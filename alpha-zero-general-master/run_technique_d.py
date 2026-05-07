import sys
import os
import argparse

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import torch
from parallel_coach import ParallelCoach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn, args as nnet_args
from utils import dotdict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import profiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/alphazero_project/checkpoints')
    args_cli = parser.parse_args()

    checkpoint_dir = args_cli.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    log.info('CUDA available: %s', torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info('CUDA device: %s', torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # Technique D uses the same stronger GPU-backed evaluation settings as Technique C.
    nnet_args.epochs = 15
    nnet_args.batch_size = 128

    # Reset profiler metrics
    profiler.iteration_metrics_list = []
    profiler.gpu_utilization_list = []
    profiler.mcts_sims_per_sec_list = []
    profiler.peak_ram_list = []
    profiler.cache_hit_rate_per_iter = []
    profiler.gpu_calls_per_iter = []
    profiler.avg_gpu_batch_size = []
    profiler.worker_utilization = []
    profiler.examples_per_worker = []
    profiler.win_rate_vs_greedy = 0.0

    args = dotdict({
        'numIters': 5,
        'numEps': 48,
        'tempThreshold': 15,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 35,
        'arenaCompare': 40,
        'greedyCompare': 40,
        'cpuct': 1,
        'checkpoint': checkpoint_dir,
        'load_model': False,
        'load_folder_file': (checkpoint_dir, 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
    })

    log.info('Loading %s...', Game.__name__)
    game = Game(6)
    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)
    log.info('Loading the ParallelCoach...')
    coach = ParallelCoach(game, nnet, args)
    log.info('Starting Technique D training (2-process parallel batched self-play)')
    coach.learn()

    metrics = profiler.save_metrics(checkpoint_dir, filename="technique_d_metrics.json")
    log.info('Technique D metrics saved: %s', metrics)


if __name__ == "__main__":
    main()
