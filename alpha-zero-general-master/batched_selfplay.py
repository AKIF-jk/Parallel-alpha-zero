import logging

import numpy as np
import torch
from tqdm import tqdm

from MCTS import EPS, MCTS, MCTSNode

try:
    import profiler
except ImportError:
    profiler = None

log = logging.getLogger(__name__)

BATCH_SIZE = 16
MAX_STEPS = 200


class BatchedSelfPlayWorker:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.num_actions = self.game.getActionSize()
        self.nn_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def execute_batch(self, num_games):
        """
        Run num_games games using lockstep MCTS batching.
        Returns list of training examples from all completed games.
        """
        all_examples = []

        for start in tqdm(range(0, num_games, BATCH_SIZE), desc="Batched Self Play"):
            wave_size = min(BATCH_SIZE, num_games - start)
            all_examples.extend(self._execute_wave(wave_size))

        return all_examples

    def cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate_pct": hit_rate * 100,
            "cache_size": len(self.nn_cache),
        }

    def _execute_wave(self, num_games):
        boards = [self.game.getInitBoard() for _ in range(num_games)]
        cur_players = [1] * num_games
        game_examples = [[] for _ in range(num_games)]
        active = [True] * num_games
        episode_steps = [0] * num_games
        mcts_list = [MCTS(self.game, self.nnet, self.args) for _ in range(num_games)]

        for mcts in mcts_list:
            mcts.nn_cache = self.nn_cache

        all_examples = []

        while any(active):
            for sim in range(self.args.numMCTSSims):
                pending = []

                for i, is_active in enumerate(active):
                    if not is_active:
                        continue

                    if profiler is not None:
                        profiler.increment_mcts_sim()

                    canonical_board = self.game.getCanonicalForm(boards[i], cur_players[i])
                    request = self._run_until_leaf(mcts_list[i], canonical_board)
                    if request is not None:
                        request["game_id"] = i
                        pending.append(request)

                if pending:
                    batch_pi, batch_v = self._batched_predict([request["board"] for request in pending])
                    if profiler is not None:
                        profiler.record_gpu_batch(len(pending))

                    for idx, request in enumerate(pending):
                        self._complete_leaf(request, batch_pi[idx], batch_v[idx])

            for i in range(num_games):
                if not active[i]:
                    continue

                episode_steps[i] += 1
                canonical_board = self.game.getCanonicalForm(boards[i], cur_players[i])
                temp = int(episode_steps[i] < self.args.tempThreshold)
                pi = self._get_action_prob(mcts_list[i], canonical_board, temp=temp)

                sym = self.game.getSymmetries(canonical_board, pi)
                for board, policy in sym:
                    game_examples[i].append([board, cur_players[i], policy, None])

                action = np.random.choice(len(pi), p=pi)
                boards[i], cur_players[i] = self.game.getNextState(boards[i], cur_players[i], action)

                result = self.game.getGameEnded(boards[i], cur_players[i])
                if result != 0:
                    all_examples.extend(self._finalize_examples(game_examples[i], result, cur_players[i]))
                    active[i] = False
                elif episode_steps[i] >= MAX_STEPS:
                    log.warning("Max self-play steps reached for game %s; forcing draw target.", i)
                    all_examples.extend(self._finalize_examples(game_examples[i], 1e-4, cur_players[i]))
                    active[i] = False

        return all_examples

    def _run_until_leaf(self, mcts, canonical_board):
        board = canonical_board
        path = []

        while True:
            s = self.game.stringRepresentation(board)
            node = mcts.nodes.get(s)

            if node is not None:
                if node.is_terminal:
                    return_value = -node.terminal_value
                    self._backup_path(path, return_value)
                    return None

                action = self._select_action(mcts, node)
                path.append((node, action))
                next_board, next_player = self.game.getNextState(board, 1, action)
                board = self.game.getCanonicalForm(next_board, next_player)
                continue

            terminal_value = self.game.getGameEnded(board, 1)
            if terminal_value != 0:
                node = MCTSNode(self.num_actions)
                node.is_terminal = True
                node.terminal_value = terminal_value
                mcts.nodes[s] = node
                self._backup_path(path, -terminal_value)
                return None

            if s in self.nn_cache:
                pi, value = self.nn_cache[s]
                self.cache_hits += 1
                self._expand_leaf(mcts, board, s, pi)
                self._backup_path(path, -value)
                return None

            return {
                "mcts": mcts,
                "board": board,
                "board_string": s,
                "path": path,
            }

    def _complete_leaf(self, request, pi, value):
        value = float(np.ravel(value)[0])
        self.cache_misses += 1

        sym = self.game.getSymmetries(request["board"], pi)
        for sym_board, sym_pi in sym:
            sym_s = self.game.stringRepresentation(sym_board)
            if sym_s not in self.nn_cache:
                self.nn_cache[sym_s] = (np.array(sym_pi), value)

        self._expand_leaf(request["mcts"], request["board"], request["board_string"], pi)
        self._backup_path(request["path"], -value)

    def _expand_leaf(self, mcts, board, board_string, pi):
        valids = self.game.getValidMoves(board, 1)

        pi = np.array(pi, dtype=np.float64) * valids
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        else:
            log.error("All valid moves were masked, doing a workaround.")
            pi = pi + valids
            pi /= np.sum(pi)

        node = MCTSNode(self.num_actions)
        node.P = pi.astype(np.float32)
        node.valid_moves = valids.astype(np.float32)
        mcts.nodes[board_string] = node

    def _backup_path(self, path, return_value):
        value = return_value
        for node, action in reversed(path):
            node.N[action] += 1
            old_n = node.N[action] - 1
            node.Q[action] = (old_n * node.Q[action] + value) / node.N[action]
            node.N_total += 1
            value = -value

    def _select_action(self, mcts, node):
        ucb = node.Q + self.args.cpuct * node.P * np.sqrt(node.N_total + EPS) / (1 + node.N)
        ucb[node.valid_moves == 0] = -np.inf
        return int(np.argmax(ucb))

    def _get_action_prob(self, mcts, canonical_board, temp=1):
        s = self.game.stringRepresentation(canonical_board)
        node = mcts.nodes.get(s)
        counts = node.N.copy() if node is not None else np.zeros(self.num_actions, dtype=np.int32)

        if temp == 0:
            best_actions = np.argwhere(counts == np.max(counts)).flatten()
            best_action = np.random.choice(best_actions)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs.tolist()

        counts = counts ** (1.0 / temp)
        counts_sum = counts.sum()
        if counts_sum == 0:
            valids = self.game.getValidMoves(canonical_board, 1)
            return (valids / np.sum(valids)).tolist()

        return (counts / counts_sum).tolist()

    def _batched_predict(self, boards):
        model = getattr(self.nnet, "model", None)
        if model is None:
            model = getattr(self.nnet, "nnet")

        device = next(model.parameters()).device
        batch = torch.FloatTensor(np.array(boards).astype(np.float64)).to(device)

        model.eval()
        with torch.no_grad():
            batch_pi, batch_v = model(batch)

        return torch.exp(batch_pi).data.cpu().numpy(), batch_v.data.cpu().numpy().reshape(-1)

    def _finalize_examples(self, examples, result, cur_player):
        return [(x[0], x[2], result * ((-1) ** (x[1] != cur_player))) for x in examples]
