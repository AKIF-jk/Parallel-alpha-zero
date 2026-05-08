from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .OthelloLogic import Board
from optimization_utils import ZobristHash
import numpy as np

try:
    import othello_bitboard
except ImportError:
    othello_bitboard = None


class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n, use_zobrist=True, zobrist_seed=42, use_bitboard=False):
        self.n = n
        self.use_zobrist = bool(use_zobrist)
        self.zobrist_hash = ZobristHash(board_size=n, seed=zobrist_seed) if self.use_zobrist else None
        self.use_bitboard = bool(use_bitboard) and n == 8 and othello_bitboard is not None

    def _new_bitboard(self, board=None):
        bb = othello_bitboard.BitBoard(self.n)
        if board is not None:
            bb.from_numpy(np.asarray(board, dtype=np.int8))
        return bb

    def getInitBoard(self):
        # return initial board (numpy board)
        if self.use_bitboard:
            bb = self._new_bitboard()
            return np.asarray(bb.to_numpy(), dtype=np.int8)
        b = Board(self.n)
        return np.array(b.pieces, dtype=np.int8)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        if self.use_bitboard:
            bb = self._new_bitboard(board)
            move = (int(action/self.n), action%self.n)
            bb.execute_move(move[0], move[1], player)
            return (np.asarray(bb.to_numpy(), dtype=np.int8), -player)
        b = Board(self.n)
        b.pieces = np.array(board, copy=True, dtype=np.int8)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        if self.use_bitboard:
            bb = self._new_bitboard(board)
            legalMoves = bb.get_legal_moves_list(player)
            if len(legalMoves)==0:
                valids[-1]=1
                return np.array(valids)
            for x, y in legalMoves:
                valids[self.n*x+y]=1
            return np.array(valids)
        b = Board(self.n)
        b.pieces = board
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if self.use_bitboard:
            bb = self._new_bitboard(board)
            if bb.has_legal_moves(player):
                return 0
            if bb.has_legal_moves(-player):
                return 0
            if bb.count_diff(player) > 0:
                return 1
            return -1
        b = Board(self.n)
        b.pieces = board
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        if self.use_bitboard:
            bb = self._new_bitboard(board)
            return int(bb.hash())
        if self.zobrist_hash is None:
            return board.tobytes()
        return self.zobrist_hash.hash_board(board)

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        if self.use_bitboard:
            bb = self._new_bitboard(board)
            return bb.count_diff(player)
        b = Board(self.n)
        b.pieces = board
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
