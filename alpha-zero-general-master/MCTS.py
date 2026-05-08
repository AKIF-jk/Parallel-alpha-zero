import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from optimization_utils import LRUCache, NumpyArrayPool

EPS = 1e-8

try:
    from profiler import increment_mcts_sim
except ImportError:
    def increment_mcts_sim():
        pass

log = logging.getLogger(__name__)


class MCTSNode:
    def __init__(self, num_actions):
        self.Q = np.zeros(num_actions, dtype=np.float32)
        self.N = np.zeros(num_actions, dtype=np.int32)
        self.N_total = 0
        self.P = np.zeros(num_actions, dtype=np.float32)
        self.valid_moves = None
        self.is_terminal = False
        self.terminal_value = 0.0
        self.virtual_loss = np.zeros(num_actions, dtype=np.int32)

        self.node_lock = threading.Lock()
        self.action_locks = [threading.Lock() for _ in range(num_actions)]
        self.expand_lock = threading.Lock()
        self.is_expanded = False


class MCTS:
    """MCTS with optional thread-pool parallel simulations (Phase 2)."""

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.num_actions = self.game.getActionSize()
        self.num_threads = int(getattr(self.args, "numMCTSThreads", 1))
        if self.num_threads <= 0:
            self.num_threads = max(1, (os.cpu_count() or 2) - 1)
        self._executor = ThreadPoolExecutor(max_workers=self.num_threads) if self.num_threads > 1 else None
        self.nodes = {}
        self.nodes_lock = threading.Lock()
        self.cache_lock = threading.Lock()

        cache_capacity = int(getattr(self.args, "nnCacheMaxSize", 500000))
        self.nn_cache = LRUCache(capacity=cache_capacity)
        action_pool_size = int(getattr(self.args, "actionArrayPoolSize", 256))
        self._action_counts_pool = NumpyArrayPool(
            shape=(self.num_actions,),
            dtype=np.float64,
            pool_size=action_pool_size,
        )
        self.cache_hits = 0
        self.cache_misses = 0

    def reset_search_tree(self):
        with self.nodes_lock:
            self.nodes = {}

    def __del__(self):
        if getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate_pct": hit_rate * 100,
            "cache_size": len(self.nn_cache),
        }

    def getActionProb(self, canonicalBoard, temp=1):
        total_sims = int(self.args.numMCTSSims)
        if self.num_threads <= 1 or total_sims <= 1:
            for _ in range(total_sims):
                increment_mcts_sim()
                self.search(canonicalBoard)
        else:
            sims_per_thread = total_sims // self.num_threads
            remainder = total_sims % self.num_threads
            futures = []
            for i in range(self.num_threads):
                sims = sims_per_thread + (1 if i < remainder else 0)
                if sims <= 0:
                    continue
                futures.append(self._executor.submit(self._search_batch, canonicalBoard, sims))
            for future in as_completed(futures):
                future.result()

        s = self.game.stringRepresentation(canonicalBoard)
        with self.nodes_lock:
            node = self.nodes.get(s)
        counts = self._action_counts_pool.acquire()
        if node is None:
            counts.fill(0)
        else:
            with node.node_lock:
                np.copyto(counts, node.N, casting="unsafe")

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            self._action_counts_pool.release(counts)
            return probs.tolist()

        np.power(counts, 1.0 / temp, out=counts)
        counts_sum = float(counts.sum())
        if counts_sum == 0:
            self._action_counts_pool.release(counts)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            return (valids / np.sum(valids)).tolist()
        probs = counts / counts_sum
        self._action_counts_pool.release(counts)
        return probs.tolist()

    def _search_batch(self, canonicalBoard, num_sims):
        for _ in range(num_sims):
            increment_mcts_sim()
            self.search(canonicalBoard)

    def _get_or_create_node(self, s):
        with self.nodes_lock:
            node = self.nodes.get(s)
            if node is None:
                node = MCTSNode(self.num_actions)
                self.nodes[s] = node
            return node

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)
        node = self._get_or_create_node(s)

        if node.is_terminal:
            return -node.terminal_value

        if not node.is_expanded:
            with node.expand_lock:
                if not node.is_expanded and not node.is_terminal:
                    terminal_value = self.game.getGameEnded(canonicalBoard, 1)
                    if terminal_value != 0:
                        node.is_terminal = True
                        node.terminal_value = terminal_value
                        return -terminal_value

                    with self.cache_lock:
                        cached = self.nn_cache.get(s)
                    if cached is not None:
                        ps, v = cached
                        self.cache_hits += 1
                    else:
                        ps, v = self.nnet.predict(canonicalBoard)
                        v = float(v)
                        with self.cache_lock:
                            sym = self.game.getSymmetries(canonicalBoard, ps)
                            for sym_board, sym_ps in sym:
                                sym_s = self.game.stringRepresentation(sym_board)
                                if self.nn_cache.get(sym_s) is None:
                                    self.nn_cache[sym_s] = (np.array(sym_ps), v)
                        self.cache_misses += 1

                    valids = self.game.getValidMoves(canonicalBoard, 1)
                    ps = ps * valids
                    sum_ps = np.sum(ps)
                    if sum_ps > 0:
                        ps /= sum_ps
                    else:
                        log.error("All valid moves were masked, doing a workaround.")
                        ps = ps + valids
                        ps /= np.sum(ps)

                    with node.node_lock:
                        node.P = ps.astype(np.float32)
                        node.valid_moves = valids.astype(np.float32)
                        node.is_expanded = True
                    return -v

        with node.node_lock:
            effective_Q = node.Q - node.virtual_loss * 1.0
            ucb = effective_Q + self.args.cpuct * node.P * math.sqrt(node.N_total + EPS) / (1 + node.N + node.virtual_loss)
            ucb[node.valid_moves == 0] = -np.inf
            a = int(np.argmax(ucb))
            node.virtual_loss[a] += 1

        next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_canonical = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_canonical)

        with node.action_locks[a]:
            with node.node_lock:
                node.virtual_loss[a] = max(0, node.virtual_loss[a] - 1)
                old_n = node.N[a]
                node.N[a] += 1
                node.Q[a] = (old_n * node.Q[a] + v) / node.N[a]
                node.N_total += 1

        return -v
