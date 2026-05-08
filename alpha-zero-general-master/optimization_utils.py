from collections import OrderedDict, deque

import numpy as np


class LRUCache:
    """Bounded LRU cache with dict-like access."""

    def __init__(self, capacity=100000):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._cache = OrderedDict()

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, key):
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def get(self, key, default=None):
        if key not in self._cache:
            return default
        self._cache.move_to_end(key)
        return self._cache[key]

    def setdefault(self, key, default):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self[key] = default
        return default

    def clear(self):
        self._cache.clear()


class NumpyArrayPool:
    """Reusable pool for fixed-shape numpy arrays."""

    def __init__(self, shape, dtype=np.float32, pool_size=256):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.pool_size = int(pool_size)
        self._available = deque(
            np.zeros(self.shape, dtype=self.dtype) for _ in range(self.pool_size)
        )

    def acquire(self):
        if self._available:
            arr = self._available.pop()
            arr.fill(0)
            return arr
        return np.zeros(self.shape, dtype=self.dtype)

    def release(self, arr):
        if not isinstance(arr, np.ndarray):
            return
        if arr.shape != self.shape or arr.dtype != self.dtype:
            return
        if len(self._available) >= self.pool_size:
            return
        self._available.append(arr)


class ZobristHash:
    """Zobrist hashing for board states with values in {-1, 0, 1}."""

    def __init__(self, board_size, num_piece_types=3, seed=42):
        self.board_size = int(board_size)
        self.num_piece_types = int(num_piece_types)
        if self.board_size <= 0:
            raise ValueError("board_size must be > 0")
        if self.num_piece_types < 3:
            raise ValueError("num_piece_types must be >= 3")

        rng = np.random.default_rng(seed)
        self.table = rng.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=(self.board_size, self.board_size, self.num_piece_types),
            dtype=np.uint64,
        )
        self._x_idx = np.arange(self.board_size)[:, None]
        self._y_idx = np.arange(self.board_size)[None, :]

    def hash_board(self, board):
        board_arr = np.asarray(board, dtype=np.int8)
        if board_arr.shape != (self.board_size, self.board_size):
            raise ValueError(
                f"board shape must be {(self.board_size, self.board_size)}, got {board_arr.shape}"
            )
        piece_idx = board_arr + 1
        selected = self.table[self._x_idx, self._y_idx, piece_idx]
        return int(np.bitwise_xor.reduce(selected.ravel()))

    def update_hash(self, current_hash, x, y, old_piece, new_piece):
        hash_val = np.uint64(current_hash)
        hash_val ^= self.table[int(x), int(y), int(old_piece) + 1]
        hash_val ^= self.table[int(x), int(y), int(new_piece) + 1]
        return int(hash_val)

    def batch_hash(self, boards):
        boards_arr = np.asarray(boards, dtype=np.int8)
        hashes = np.zeros(boards_arr.shape[0], dtype=np.uint64)
        for idx, board in enumerate(boards_arr):
            hashes[idx] = np.uint64(self.hash_board(board))
        return hashes
