import logging
import os
import queue
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
import torch.multiprocessing as mp

from Arena import Arena
from batched_selfplay import BatchedSelfPlayWorker
from MCTS import MCTS

log = logging.getLogger(__name__)

try:
    import profiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


def _model_module(nnet):
    model = getattr(nnet, "model", None)
    if model is None:
        model = getattr(nnet, "nnet")
    return model


class QueuePredictor:
    """Worker-side synchronous RPC client for main-process NN inference."""

    def __init__(self, worker_id, prediction_queue, response_queue):
        self.worker_id = worker_id
        self.prediction_queue = prediction_queue
        self.response_queue = response_queue
        self.request_id = 0

    def __call__(self, boards):
        self.request_id += 1
        boards = np.asarray(boards, dtype=np.float32)
        self.prediction_queue.put((self.worker_id, self.request_id, boards))
        response_id, policies, values = self.response_queue.get()
        if response_id != self.request_id:
            raise RuntimeError(
                f"Worker {self.worker_id} received stale inference response "
                f"{response_id}; expected {self.request_id}"
            )
        return policies, values


def worker_fn(worker_id, game, args, prediction_queue, response_queue, result_queue, num_games):
    """Function run in each worker process."""
    torch.set_num_threads(1)
    started_at = time.perf_counter()

    try:
        predictor = QueuePredictor(worker_id, prediction_queue, response_queue)
        worker = BatchedSelfPlayWorker(game, None, args, predictor=predictor)
        examples = worker.execute_batch(num_games)

        elapsed = time.perf_counter() - started_at
        cache_stats = worker.cache_stats()
        batch_stats = worker.batch_stats()
        result_queue.put((
            worker_id,
            examples,
            {
                "elapsed_sec": elapsed,
                "example_count": len(examples),
                "cache_stats": cache_stats,
                "batch_stats": batch_stats,
                "num_games": num_games,
            },
            None,
        ))
    except Exception:
        import traceback
        result_queue.put((worker_id, [], {}, traceback.format_exc()))


def _batched_model_predict(nnet, boards):
    model = _model_module(nnet)
    device = next(model.parameters()).device
    batch = torch.as_tensor(np.asarray(boards, dtype=np.float32), device=device)

    model.eval()
    with torch.no_grad():
        batch_pi, batch_v = model(batch)

    return torch.exp(batch_pi).data.cpu().numpy(), batch_v.data.cpu().numpy().reshape(-1)


def _serve_prediction_requests(
    first_request,
    prediction_queue,
    response_queues,
    game,
    nnet,
    args,
    stats,
    inference_cache,
):
    """
    Coalesce worker inference requests into one main-process model call.

    Workers do CPU-side MCTS and block here only for leaf evaluation. Keeping
    CUDA in the parent process avoids multiple CUDA contexts contending for the
    same GPU and lets requests from all workers form one larger batch.
    """
    requests = [first_request]
    wait_sec = float(getattr(args, "inferenceBatchWaitSec", 0.003))
    deadline = time.perf_counter() + wait_sec

    while time.perf_counter() < deadline:
        timeout = max(0.0, deadline - time.perf_counter())
        try:
            requests.append(prediction_queue.get(timeout=timeout))
        except queue.Empty:
            break

    unique_boards = []
    unique_keys = []
    unique_index = {}
    placements = []

    for worker_id, request_id, boards in requests:
        request_positions = []
        for board in boards:
            board = np.asarray(board, dtype=np.float32)
            key = game.stringRepresentation(board)
            cached = inference_cache.get(key)
            if cached is not None:
                stats["cache_hits"] += 1
                request_positions.append(("cache", cached[0], cached[1]))
                continue

            idx = unique_index.get(key)
            if idx is None:
                idx = len(unique_boards)
                unique_index[key] = idx
                unique_boards.append(board)
                unique_keys.append(key)
            request_positions.append(("gpu", idx))
        placements.append((worker_id, request_id, request_positions))

    if unique_boards:
        batch_pi, batch_v = _batched_model_predict(nnet, unique_boards)
        stats["gpu_calls"] += 1
        stats["boards_to_gpu"] += len(unique_boards)
        stats["cache_misses"] += len(unique_boards)

        for idx, board in enumerate(unique_boards):
            pi = np.asarray(batch_pi[idx], dtype=np.float32)
            value = float(batch_v[idx])
            inference_cache.setdefault(unique_keys[idx], (pi, value))
            for sym_board, sym_pi in game.getSymmetries(board, pi):
                sym_key = game.stringRepresentation(np.asarray(sym_board, dtype=np.float32))
                inference_cache.setdefault(
                    sym_key,
                    (np.asarray(sym_pi, dtype=np.float32), value),
                )
    else:
        batch_pi = np.empty((0, 0), dtype=np.float32)
        batch_v = np.empty((0,), dtype=np.float32)

    for worker_id, request_id, request_positions in placements:
        policies = []
        values = []
        for source, payload, *rest in request_positions:
            if source == "cache":
                policies.append(payload)
                values.append(rest[0])
            else:
                idx = payload
                policies.append(batch_pi[idx])
                values.append(batch_v[idx])
        policies = np.asarray(policies, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        response_queues[worker_id].put((request_id, policies, values))


class ParallelCoach:
    """
    Coach variant for Technique D: spawned CPU self-play workers with
    main-process neural inference.

    Workers generate examples and run MCTS traversal. The main process owns the
    PyTorch model during self-play, batches requests across workers, and keeps
    training, arena comparison, checkpointing, and model acceptance/rejection
    in one process.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

    def learn(self):
        for iteration in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{iteration} ...')

            if PROFILER_AVAILABLE:
                profiler.reset_mcts_sim()
                profiler.reset_gpu_batch_stats()
                profiler.start_gpu_monitor()
                profiler.start_phase("self_play")
                iteration_start_ram = profiler.get_peak_ram_mb()

            cache_stats = {"hits": 0, "misses": 0, "hit_rate_pct": 0.0, "cache_size": 0}
            avg_batch_size = 0.0
            gpu_calls = 0
            total_mcts = 0
            virtual_loss_diversions = 0
            worker_elapsed = []
            num_workers = int(getattr(self.args, "numWorkers", 2))
            worker_example_counts = [0] * num_workers

            if not self.skipFirstSelfPlay or iteration > 1:
                base_games = self.args.numEps // num_workers
                remainder = self.args.numEps % num_workers
                result_queue = mp.Queue()
                prediction_queue = mp.Queue()
                response_queues = [mp.Queue() for _ in range(num_workers)]
                processes = []
                for worker_id in range(num_workers):
                    num_games = base_games + (1 if worker_id < remainder else 0)
                    process = mp.Process(
                        target=worker_fn,
                        args=(
                            worker_id,
                            self.game,
                            self.args,
                            prediction_queue,
                            response_queues[worker_id],
                            result_queue,
                            num_games,
                        ),
                    )
                    process.start()
                    processes.append(process)

                worker_results = []
                inference_stats = {
                    "gpu_calls": 0,
                    "boards_to_gpu": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                }
                inference_cache = {}
                while len(worker_results) < num_workers:
                    try:
                        result = result_queue.get_nowait()
                        worker_id, examples, stats, tb = result
                        if tb:
                            raise RuntimeError(f"Self-play worker {worker_id} failed:\n{tb}")
                        worker_results.append((worker_id, examples, stats))
                        continue
                    except queue.Empty:
                        pass

                    try:
                        request = prediction_queue.get(timeout=0.1)
                    except queue.Empty:
                        dead = [p.exitcode for p in processes if not p.is_alive() and p.exitcode not in (0, None)]
                        if dead:
                            raise RuntimeError(f"Self-play worker exited unexpectedly: {dead}")
                        continue

                    _serve_prediction_requests(
                        request,
                        prediction_queue,
                        response_queues,
                        self.game,
                        self.nnet,
                        self.args,
                        inference_stats,
                        inference_cache,
                    )

                for process in processes:
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"Self-play worker exited with code {process.exitcode}")

                all_examples = []
                cache_hits = 0
                cache_misses = 0
                cache_size = 0
                total_boards_to_gpu = 0

                for worker_id, examples, stats in worker_results:
                    all_examples.extend(examples)
                    worker_example_counts[worker_id] = stats["example_count"]
                    worker_elapsed.append(stats["elapsed_sec"])

                    worker_cache = stats["cache_stats"]
                    cache_hits += worker_cache["hits"]
                    cache_misses += worker_cache["misses"]
                    cache_size += worker_cache["cache_size"]

                    worker_batch = stats["batch_stats"]
                    total_mcts += worker_batch["mcts_sim_count"]
                    virtual_loss_diversions += worker_batch.get("virtual_loss_diversions", 0)

                total_cache = cache_hits + cache_misses
                cache_stats = {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "hit_rate_pct": (cache_hits / total_cache * 100) if total_cache > 0 else 0.0,
                    "cache_size": cache_size,
                }
                gpu_calls = inference_stats["gpu_calls"]
                total_boards_to_gpu = inference_stats["boards_to_gpu"]
                avg_batch_size = total_boards_to_gpu / gpu_calls if gpu_calls > 0 else 0.0
                avg_vl_collisions = virtual_loss_diversions / total_mcts if total_mcts > 0 else 0.0

                iterationTrainExamples = deque(all_examples, maxlen=self.args.maxlenOfQueue)
                self.trainExamplesHistory.append(iterationTrainExamples)
            else:
                avg_vl_collisions = 0.0

            if PROFILER_AVAILABLE:
                self_play_sec = profiler.end_phase("self_play")
                peak_ram = max(iteration_start_ram, profiler.get_peak_ram_mb())
                worker_util = (
                    sum(worker_elapsed) / (num_workers * self_play_sec) * 100
                    if self_play_sec > 0 and worker_elapsed else 0.0
                )
                profiler.start_phase("train")
            else:
                self_play_sec = 0.0
                peak_ram = 0.0
                worker_util = 0.0

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(iteration - 1)

            trainExamples = []
            for examples in self.trainExamplesHistory:
                trainExamples.extend(examples)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            if PROFILER_AVAILABLE:
                train_sec = profiler.end_phase("train")
                peak_ram = max(peak_ram, profiler.get_peak_ram_mb())
                profiler.start_phase("arena")
            else:
                train_sec = 0.0

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            self.nnet.save_checkpoint(
                folder=self.args.checkpoint,
                filename=f'iter_{iteration:03d}.pth.tar'
            )

            if PROFILER_AVAILABLE:
                arena_sec = profiler.end_phase("arena")
                peak_ram = max(peak_ram, profiler.get_peak_ram_mb())
                profiler.stop_gpu_monitor()

                mcts_sps = total_mcts / self_play_sec if self_play_sec > 0 else 0.0
                profiler.iteration_metrics_list.append({
                    "self_play_sec": self_play_sec,
                    "train_sec": train_sec,
                    "arena_sec": arena_sec
                })
                profiler.mcts_sims_per_sec_list.append(mcts_sps)
                profiler.peak_ram_list.append(peak_ram)
                profiler.cache_hit_rate_per_iter.append(cache_stats["hit_rate_pct"])
                profiler.gpu_calls_per_iter.append(gpu_calls)
                profiler.avg_gpu_batch_size.append(avg_batch_size)
                profiler.avg_virtual_loss_collisions_avoided.append(avg_vl_collisions)
                profiler.worker_utilization.append(worker_util)
                profiler.examples_per_worker.append(worker_example_counts)

            if iteration == 5 and PROFILER_AVAILABLE:
                log.info("Running win rate vs greedy baseline...")
                from othello.OthelloPlayers import GreedyOthelloPlayer
                greedy_player = GreedyOthelloPlayer(self.game)
                greedy_compare = getattr(self.args, "greedyCompare", 20)

                def nnet_player(canonicalBoard):
                    mcts = MCTS(self.game, self.nnet, self.args)
                    return np.argmax(mcts.getActionProb(canonicalBoard, temp=0))

                arena_greedy = Arena(nnet_player, greedy_player.play, self.game)
                nwins, gwins, draws = arena_greedy.playGames(greedy_compare)
                profiler.win_rate_vs_greedy = nwins / greedy_compare
                log.info(f"Win rate vs greedy: {profiler.win_rate_vs_greedy}")

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            response = input("Continue? [y|n]")
            if response != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
            self.skipFirstSelfPlay = True
