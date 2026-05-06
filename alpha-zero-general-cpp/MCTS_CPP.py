"""
C++-backed MCTS implementation. Drop-in replacement for MCTS.py.
Uses the C++ MCTS tree (via pybind11) for fast tree traversal while
calling the Python neural network for policy/value predictions.
"""
import logging
import math

import numpy as np
import othello_cpp

log = logging.getLogger(__name__)


class MCTS_CPP:
    """MCTS using C++ tree for fast traversal, Python NN for predictions."""

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        # Cache for policy/value of board states we've seen in this tree
        # Key: board bytes, Value: (policy_vec, value)
        self._cache = {}

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Performs numMCTSSims simulations using C++ MCTS tree.

        Returns:
            probs: policy vector of length game.getActionSize()
        """
        self._cache.clear()

        # Set root board state for C++ MCTS
        board_flat = canonicalBoard.flatten().astype(np.int8).tolist()
        self.cpp_mcts = othello_cpp.BatchedMCTS()
        self.cpp_mcts.set_root_board(board_flat, 1)

        # Run MCTS simulations
        for _ in range(self.args.numMCTSSims):
            self._search(canonicalBoard)

        # Extract policy from root visit counts
        counts = self.cpp_mcts.get_root_visit_counts()
        counts = np.array(counts[:self.game.getActionSize()], dtype=np.float64)

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(self.game.getActionSize())
            probs[bestA] = 1.0
            return probs

        counts = np.power(counts, 1.0 / temp)
        counts_sum = float(np.sum(counts))
        if counts_sum > 0:
            probs = counts / counts_sum
        else:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            probs = valids / np.sum(valids)

        return probs

    def _search(self, root_board):
        """Single MCTS simulation using C++ tree + Python NN prediction."""
        leaf_idx, state_tensor, legal_moves, is_terminal, val, hit, tt_p, tt_v = \
            self.cpp_mcts.select_and_get_leaf()

        if is_terminal:
            self.cpp_mcts.expand_and_backup(leaf_idx, legal_moves, [0.0]*36, float(val))
            return -val

        if hit:
            policy = list(tt_p)
            self.cpp_mcts.expand_and_backup(leaf_idx, legal_moves, policy, float(tt_v))
            return -tt_v

        # Get valid moves mask for this board size
        action_size = self.game.getActionSize()
        valids = np.zeros(action_size)
        for m in legal_moves:
            if 0 <= m < action_size:
                valids[m] = 1
        if np.sum(valids) == 0:
            valids[-1] = 1

        # Use the state tensor from select_and_get_leaf() - this is the board at the leaf
        # State tensor is 2x6x6: plane 0 = current player pieces, plane 1 = opponent pieces
        state_np = np.array(state_tensor).reshape(2, 6, 6)
        
        # Get current player from C++ tree
        current_player = self.cpp_mcts.get_current_player()
        
        # Reconstruct canonical board (player 1 perspective):
        # If current_player is 1: board = plane_0 - plane_1 (player1 pieces - opponent)
        # If current_player is -1: board = plane_1 - plane_0 (flip perspective)
        if current_player == 1:
            canonical = state_np[0] - state_np[1]
        else:
            canonical = state_np[1] - state_np[0]

        # Neural network prediction
        pi, v = self.nnet.predict(canonical)

        # Mask invalid moves and renormalize
        pi = pi * valids
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        else:
            pi = valids / np.sum(valids)

        # Expand and backup
        v_scalar = float(v) if v.ndim == 0 else float(v[0])
        self.cpp_mcts.expand_and_backup(leaf_idx, legal_moves, pi.tolist(), v_scalar)
        return -v_scalar
