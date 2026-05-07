import logging
import math

import numpy as np

EPS = 1e-8

try:
    from profiler import increment_mcts_sim
except ImportError:
    def increment_mcts_sim():
        pass

log = logging.getLogger(__name__)


class MCTSNode:
    def __init__(self, num_actions):
        self.Q = np.zeros(num_actions, dtype=np.float32)  # Q values per action
        self.N = np.zeros(num_actions, dtype=np.int32)     # visit count per action
        self.N_total = 0                                    # total visits to this node
        self.P = np.zeros(num_actions, dtype=np.float32)   # prior probs
        self.valid_moves = None                             # valid moves mask (1=valid, 0=invalid)
        self.is_terminal = False
        self.terminal_value = 0.0
        self.virtual_loss = np.zeros(num_actions, dtype=np.int32)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.num_actions = self.game.getActionSize()
        self.nodes = {}  # maps board string s -> MCTSNode
        self.nn_cache = {}  # board_string -> (policy_np_array, value_float)
        self.cache_hits = 0
        self.cache_misses = 0

    def reset_search_tree(self):
        self.nodes = {}

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
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            increment_mcts_sim()  # Count only top-level simulations
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        node = self.nodes.get(s)
        if node is None:
            counts = np.zeros(self.num_actions, dtype=np.int32)
        else:
            counts = node.N.copy()  # per-action visit counts

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros_like(counts)
            probs[bestA] = 1
            return probs.tolist()
        
        counts = counts ** (1. / temp)
        counts_sum = counts.sum()
        probs = counts / counts_sum
        return probs.tolist()

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # Check terminal state
        if s in self.nodes:
            node = self.nodes[s]
            if node.is_terminal:
                return -node.terminal_value
        else:
            terminal_value = self.game.getGameEnded(canonicalBoard, 1)
            if terminal_value != 0:
                node = MCTSNode(self.num_actions)
                node.is_terminal = True
                node.terminal_value = terminal_value
                self.nodes[s] = node
                return -terminal_value

        # Handle leaf node
        if s not in self.nodes:
            if s in self.nn_cache:
                ps, v = self.nn_cache[s]
                self.cache_hits += 1
            else:
                ps, v = self.nnet.predict(canonicalBoard)
                v = float(v)

                # Cache this network result and all symmetric equivalents.
                sym = self.game.getSymmetries(canonicalBoard, ps)
                for sym_board, sym_ps in sym:
                    sym_s = self.game.stringRepresentation(sym_board)
                    if sym_s not in self.nn_cache:
                        self.nn_cache[sym_s] = (np.array(sym_ps), v)
                self.cache_misses += 1

            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Mask invalid moves
            ps = ps * valids
            sum_ps = np.sum(ps)
            if sum_ps > 0:
                ps /= sum_ps
            else:
                log.error("All valid moves were masked, doing a workaround.")
                ps = ps + valids
                ps /= np.sum(ps)
            
            node = MCTSNode(self.num_actions)
            node.P = ps.astype(np.float32)
            node.valid_moves = valids.astype(np.float32)
            self.nodes[s] = node
            return -v

        # Compute UCB for all actions (vectorized)
        node = self.nodes[s]
        effective_Q = node.Q - node.virtual_loss * 1.0
        ucb = effective_Q + self.args.cpuct * node.P * np.sqrt(node.N_total + EPS) / (1 + node.N + node.virtual_loss)
        ucb[node.valid_moves == 0] = -np.inf  # mask invalid actions
        best_act = np.argmax(ucb)

        # Recurse to next state
        a = best_act
        next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_canonical = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_canonical)

        # Update action stats
        node.N[a] += 1
        old_N = node.N[a] - 1
        node.Q[a] = (old_N * node.Q[a] + v) / node.N[a]
        node.N_total += 1

        return -v
