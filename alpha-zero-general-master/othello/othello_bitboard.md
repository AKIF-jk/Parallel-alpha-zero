# `othello_bitboard.cpp` — Implementation Reference

> High-performance 8×8 Othello (Reversi) engine exposed to Python via **pybind11**,
> with **bitboard** game logic, **SIMD** compiler hints, and **OpenMP** batch parallelism.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Board Representation](#2-board-representation)
3. [Bit-to-Square Mapping](#3-bit-to-square-mapping)
4. [Direction Shifts](#4-direction-shifts)
5. [Core Algorithms](#5-core-algorithms)
   - 5.1 [Legal Move Generation — `legal_dir`](#51-legal-move-generation--legal_dir)
   - 5.2 [Flip Computation — `ray_flips`](#52-flip-computation--ray_flips)
6. [Public API](#6-public-api)
   - 6.1 [Single-Board Methods](#61-single-board-methods)
   - 6.2 [Batch Free Functions](#62-batch-free-functions)
7. [Performance Optimisations](#7-performance-optimisations)
   - 7.1 [Compiler Pragmas](#71-compiler-pragmas)
   - 7.2 [Hardware POPCNT](#72-hardware-popcnt)
   - 7.3 [Cache Alignment](#73-cache-alignment)
   - 7.4 [Branch Prediction Hints](#74-branch-prediction-hints)
   - 7.5 [No-Alias Pointers](#75-no-alias-pointers)
   - 7.6 [Loop Unrolling](#76-loop-unrolling)
   - 7.7 [OpenMP Parallelism](#77-openmp-parallelism)
8. [Thread Safety](#8-thread-safety)
9. [NumPy Interoperability](#9-numpy-interoperability)
10. [Building the Extension](#10-building-the-extension)
11. [Python Usage Examples](#11-python-usage-examples)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. Overview

The module implements a complete Othello game engine in a single C++ file.
The central data structure is `BitBoard`: two 64-bit integers (`black_`,
`white_`) represent the positions of all black and white discs respectively.
One bit encodes one square — making every board operation a handful of integer
instructions rather than an array traversal.

The Python-facing class `BitBoard` and two module-level batch functions are
registered with **pybind11** and compiled into a shared library importable
directly from Python.

```
othello_bitboard
├── class BitBoard          ← single-board, all original methods preserved
├── batch_get_legal_moves_mask(boards, color)   ← NEW, OpenMP-parallel
└── batch_count_diff(boards, color)             ← NEW, OpenMP-parallel
```

---

## 2. Board Representation

```
Bit index layout (row-major, 0 = top-left):

  col →   0    1    2    3    4    5    6    7
row ↓  +----+----+----+----+----+----+----+----+
  0    |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
  1    |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
  ...
  7    | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
       +----+----+----+----+----+----+----+----+
```

`bit index = row × 8 + col`

Two separate 64-bit masks are maintained — one for each colour — so checking
whether a cell is occupied is a single bitwise AND and comparing with zero.

**Convention** (matches the Python side):

| Value | Meaning |
|-------|---------|
| `white_ bit set` | White disc (`+1` in numpy) |
| `black_ bit set` | Black disc (`-1` in numpy) |
| neither bit set  | Empty cell (`0` in numpy) |

**Starting position** (standard Othello):

```cpp
white_ = (1ULL << 27) | (1ULL << 36);   // (3,3) and (4,4)
black_ = (1ULL << 28) | (1ULL << 35);   // (3,4) and (4,3)
```

---

## 3. Bit-to-Square Mapping

```
square (r, c)  →  bit  1ULL << (r*8 + c)
bit index  b   →  row  b / 8,  col  b % 8
```

Extracting all set bits from a mask is done with the classic LSB-peel loop:

```cpp
while (moves) {
    int bit = __builtin_ctzll(moves);   // index of lowest set bit (BSF)
    // ... use bit ...
    moves &= moves - 1;                 // clear lowest set bit
}
```

`__builtin_ctzll` compiles to a single `TZCNT` / `BSF` instruction on x86.

---

## 4. Direction Shifts

All eight Othello directions are expressed as bitwise shifts of the 64-bit mask.
Horizontal shifts require masking off the wrap-around column:

```
Direction   Shift          Wrap guard
─────────────────────────────────────────────────────────
North (N)   b << 8         none (rows wrap cleanly)
South (S)   b >> 8         none
East  (E)   b << 1         AND NOT_A_FILE  (mask 0xfefe…)
West  (W)   b >> 1         AND NOT_H_FILE  (mask 0x7f7f…)
NE          b << 9         AND NOT_A_FILE
NW          b << 7         AND NOT_H_FILE
SE          b >> 7         AND NOT_A_FILE
SW          b >> 9         AND NOT_H_FILE
```

The file masks prevent a disc on column H from wrapping to column A when
shifted east (and vice versa for west shifts).

```
NOT_A_FILE = 0xfefefefefefefefe  (all columns except A / col 0)
NOT_H_FILE = 0x7f7f7f7f7f7f7f7f  (all columns except H / col 7)
```

---

## 5. Core Algorithms

### 5.1 Legal Move Generation — `legal_dir`

```cpp
static uint64_t legal_dir(uint64_t player, uint64_t opponent,
                           uint64_t (*shift)(uint64_t),
                           uint64_t border_mask)
```

For a single direction, legal squares are those reachable by:
1. stepping from a player disc into a consecutive run of **one or more** opponent discs, then
2. landing on an **empty** cell.

This is computed entirely with bitmask arithmetic — no loops over squares:

```
Step 0:  x  = shift(player) & opponent & border_mask
            ↑ cells immediately adjacent to player discs (in that direction)
              that are occupied by the opponent

Step 1–5: x |= shift(x) & opponent & border_mask
            ↑ flood-fill along the opponent run (up to 6 cells, hence 6 steps)

Result:   shift(x) & empty & border_mask
            ↑ the cell beyond the run must be empty
```

Six propagation steps are sufficient because the board is 8 cells wide — a run can be at most 6 opponent discs long.

All six steps are **manually unrolled** in source code (no loop), giving the
compiler a dependency-free sequence it can schedule optimally.

### 5.2 Flip Computation — `ray_flips`

```cpp
static uint64_t ray_flips(uint64_t move,
                           uint64_t player, uint64_t opponent,
                           uint64_t (*shift)(uint64_t),
                           uint64_t border_mask)
```

Given a placed disc (`move`), walk in one direction accumulating opponent discs
until either the board edge is reached or a player disc is found:

```
cur = shift(move) & border_mask
while cur is set AND cur is an opponent disc:
    flips |= cur
    cur = shift(cur) & border_mask

if cur is a player disc → return flips   (valid sandwich)
else                    → return 0       (no flip)
```

`get_legal_moves_mask` and `execute_move` both call their respective helpers
once per direction (8 calls total), then OR the results together.

---

## 6. Public API

### 6.1 Single-Board Methods

All method signatures are **identical to the original baseline** — the Python
interface is fully backward compatible.

#### `BitBoard(size=8)`
Constructor. Only `size=8` is supported; any other value raises `RuntimeError`.
Initialises the board to the standard Othello starting position.

---

#### `get_legal_moves_list(color: int) → List[Tuple[int, int]]`
Returns a list of `(row, col)` pairs representing all legal moves for `color`.

Internally calls `get_legal_moves_mask`, then peels bits with `__builtin_ctzll`.
`vector::reserve` is called with the exact move count (from `__builtin_popcountll`)
to eliminate heap reallocations.

```python
moves = board.get_legal_moves_list(1)   # white's moves
# → [(2, 3), (3, 2), ...]
```

---

#### `has_legal_moves(color: int) → bool`
Returns `True` if the player has at least one legal move. Delegates to
`get_legal_moves_mask` and tests for non-zero — no allocation.

---

#### `execute_move(x: int, y: int, color: int) → None`
Places a disc of `color` at `(x, y)` and flips all sandwiched opponent discs.
Calls `ray_flips` in all 8 directions, ORs results, then updates `white_` and
`black_` atomically (no intermediate state exposed).

---

#### `count_diff(color: int) → int`
Returns `(player disc count) − (opponent disc count)`.
Uses `HW_POPCNT` (hardware `POPCNT` instruction on x86) for each operand —
two CPU instructions total.

---

#### `to_numpy() → np.ndarray[int8, shape=(8,8)]`
Converts the bitboard to a row-major 8×8 numpy array.
Cell values: `1` = white, `-1` = black, `0` = empty.

The inner `y`-loop is unrolled 8 ways by `#pragma GCC unroll 8`.

---

#### `from_numpy(arr: np.ndarray[int8, shape=(8,8)]) → None`
Loads board state from a numpy array (the inverse of `to_numpy`).
The array is forced to C-contiguous layout by `py::array::c_style | forcecast`.
Raises `RuntimeError` if shape is not `(8, 8)`.

---

#### `hash() → int`
A lightweight hash of the board state, suitable for transposition tables:

```cpp
return black_ ^ (white_ << 1) ^ (white_ >> 63);
```

This is **not** a cryptographic hash. Collisions are possible but rare in
practice for game-tree search.

---

### 6.2 Batch Free Functions

These are **module-level** functions (not methods), added as new entry-points.
They accept a 3-D numpy array of shape `(N, 8, 8)`, dtype `int8`, and process
all N boards in parallel using OpenMP.

#### `batch_get_legal_moves_mask(boards, color: int) → List[int]`

```python
import numpy as np
import othello_bitboard as ob

boards = np.zeros((50000, 8, 8), dtype=np.int8)
# ... fill boards ...
masks = ob.batch_get_legal_moves_mask(boards, color=1)
# masks[i] is a uint64 bitmask of legal moves for board i
```

Returns one 64-bit integer per board. To decode individual squares:

```python
import numpy as np
mask = masks[0]
squares = [(b // 8, b % 8) for b in range(64) if (mask >> b) & 1]
```

---

#### `batch_count_diff(boards, color: int) → List[int]`

```python
diffs = ob.batch_count_diff(boards, color=-1)
# diffs[i] = (black pieces - white pieces) on board i
```

Returns one signed integer per board.

---

## 7. Performance Optimisations

### 7.1 Compiler Pragmas

```cpp
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")
```

These file-scoped pragmas are honoured by both GCC and Clang. They ensure:

- Full `-O3` optimisation regardless of the compile-line flags used.
- The compiler may emit AVX2, BMI/BMI2, and `POPCNT` instructions for this
  translation unit specifically, even if the project is otherwise compiled for a
  generic target.

**Effect**: auto-vectorised loops, fused compare-and-mask, single-instruction
bit operations (`LZCNT`, `TZCNT`, `POPCNT`, `PDEP`, `PEXT`).

---

### 7.2 Hardware POPCNT

```cpp
#if defined(__x86_64__) || ...
#  define HW_POPCNT(x) static_cast<int>(_mm_popcnt_u64(x))
#else
#  define HW_POPCNT(x) static_cast<int>(__builtin_popcountll(x))
#endif
```

`_mm_popcnt_u64` is an Intel intrinsic that maps directly to the `POPCNT`
machine instruction — one cycle latency, one cycle throughput on modern Intel/AMD
cores. The compiler cannot always guarantee this when using `__builtin_popcountll`
without `-mpopcnt`, so the intrinsic is used explicitly on x86.

The fallback (`__builtin_popcountll`) is still efficient on AArch64 (mapped to
`CNT` + `ADDV`) and RISC-V with the B-extension.

**Used in**: `count_diff`, `get_legal_moves_list` (for `reserve`), both batch
helpers.

---

### 7.3 Cache Alignment

```cpp
class alignas(64) BitBoard { ... };
```

Aligning `BitBoard` to 64 bytes (one cache line) has two benefits:

1. **No false sharing** — when multiple threads each hold a private `BitBoard`
   on the stack (as in the batch loops), their data occupy distinct cache lines
   and do not invalidate each other's caches.
2. **Aligned loads** — the compiler can issue `VMOVDQA` (aligned move) instead
   of `VMOVDQU` (unaligned), which avoids cross-line penalties on older
   microarchitectures.

---

### 7.4 Branch Prediction Hints

```cpp
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
```

`__builtin_expect` communicates probability to the compiler's branch predictor
model so it lays out hot and cold code paths optimally in the instruction stream.

| Site | Annotation | Rationale |
|------|-----------|-----------|
| `ray_flips` while-condition | `LIKELY` | Most rays traverse at least one disc |
| `ray_flips` final if | `LIKELY` | Valid sandwich is the expected outcome |
| Constructor size check | `UNLIKELY` | Error path — almost never taken |
| `from_numpy` shape check | `UNLIKELY` | Error path |
| `get_legal_moves_list` while | `LIKELY` | Loop body executes most iterations |

---

### 7.5 No-Alias Pointers

```cpp
void load_raw(const int8_t * __restrict__ data)
static inline void load_board_slice(BitBoard &bb,
                                    const int8_t * __restrict__ base, ...)
```

`__restrict__` (GCC/Clang extension, equivalent to C99 `restrict`) asserts that
`data` does not alias any other pointer the function can reach. This unlocks:

- Vectorised memory loads (the compiler can widen to 128/256-bit loads).
- Loop-invariant code motion across pointer dereferences.
- Elimination of reload fences between iterations.

---

### 7.6 Loop Unrolling

```cpp
#pragma GCC unroll 8
for (int y = 0; y < 8; ++y) { ... }
```

Applied to the inner loops of `to_numpy` and `from_numpy`. Since `y` runs
exactly 8 times and each iteration is independent, full unrolling:

- Removes the branch-and-decrement overhead entirely.
- Exposes 8 independent bit-test operations the out-of-order execution unit
  can issue simultaneously.

The `legal_dir` propagation steps are also hand-unrolled (6 explicit `x |= ...`
lines) rather than a `for` loop, for the same reason.

---

### 7.7 OpenMP Parallelism

```cpp
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) default(none) \
      shared(base, result, color) firstprivate(N)
#endif
for (py::ssize_t i = 0; i < N; ++i) {
    BitBoard bb;
    load_board_slice(bb, base, i);
    result[i] = bb.get_legal_moves_mask(color);   // or count_diff
}
```

Key design choices:

| Choice | Reason |
|--------|--------|
| `schedule(dynamic, 64)` | Chunks of 64 boards balance load without excessive scheduling overhead |
| `default(none)` | Forces explicit declaration of every shared variable — prevents accidental sharing bugs |
| `firstprivate(N)` | Each thread gets its own copy of the loop bound — avoids cache-line ping-pong on a shared `N` |
| Private `BitBoard bb` on stack | Each thread constructs its own `BitBoard` — zero contention, zero mutex |
| Pre-allocated `result` vector | Writing `result[i]` from thread `i` is safe because no two threads share an index |

The `#ifdef _OPENMP` guard means the file compiles and runs correctly even when
`-fopenmp` is omitted — it degrades gracefully to single-threaded execution.

---

## 8. Thread Safety

The `BitBoard` class itself is **not** thread-safe for concurrent writes to a
single instance (no locking). This is intentional — game-tree search typically
copies board state anyway.

The batch functions are thread-safe by construction:

```
Thread 0 → private BitBoard bb0  →  result[0]
Thread 1 → private BitBoard bb1  →  result[1]
...
Thread k → private BitBoard bbk  →  result[k]
```

- The input array `base` is read-only — safe to share.
- Each thread writes to a distinct index of `result` — no race.
- No heap allocations are shared across threads (each `BitBoard` is stack-allocated).

---

## 9. NumPy Interoperability

### Single-board exchange

```
Python numpy (8,8) int8
        ↕  from_numpy / to_numpy
C++ BitBoard (black_, white_)
```

`py::array::c_style | py::array::forcecast` in `from_numpy` guarantees:

- The array is row-major (C order) — matches the bit-index layout.
- Non-contiguous or Fortran-order arrays are copied/reinterpreted automatically
  by pybind11 before the function body runs.

### Batch exchange

```
Python numpy (N, 8, 8) int8
        ↕  batch_* functions
C++ vector<uint64_t> or vector<int>
        ↕  pybind11 STL conversion
Python List[int]
```

`buf.data(0, 0, 0)` returns a raw pointer to the first element of the
C-contiguous 3-D array. `load_raw` then reads 64 consecutive bytes per board —
exactly one cache line per board on 64-byte-line hardware.

---

## 10. Building the Extension

### With OpenMP and native SIMD (recommended for production)

```bash
c++ -O3 -march=native -fopenmp -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    othello_bitboard.cpp \
    -o othello_bitboard$(python3-config --extension-suffix)
```

### Without OpenMP (single-threaded, SIMD still active)

```bash
c++ -O3 -march=native -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    othello_bitboard.cpp \
    -o othello_bitboard$(python3-config --extension-suffix)
```

### Clang equivalent

```bash
clang++ -O3 -march=native -fopenmp -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    othello_bitboard.cpp \
    -o othello_bitboard$(python3-config --extension-suffix)
```

### Disabling SIMD intrinsics (cross-compilation or non-x86)

No action required — the `#if defined(__x86_64__)` guard selects the portable
`__builtin_popcountll` fallback automatically on non-x86 targets.

---

## 11. Python Usage Examples

### Basic game loop

```python
import othello_bitboard as ob

board = ob.BitBoard()

# White's turn
moves = board.get_legal_moves_list(1)
if moves:
    r, c = moves[0]
    board.execute_move(r, c, 1)

# Inspect the board
arr = board.to_numpy()          # shape (8,8), int8
print(board.count_diff(1))      # white - black piece count
print(board.has_legal_moves(-1))  # can black move?

# Transposition table key
key = board.hash()
```

### Load and save arbitrary positions

```python
import numpy as np

state = np.array([...], dtype=np.int8).reshape(8, 8)
board.from_numpy(state)
restored = board.to_numpy()
```

### Batch evaluation (Monte Carlo / MCTS rollouts)

```python
import numpy as np
import othello_bitboard as ob

N = 100_000
boards = np.zeros((N, 8, 8), dtype=np.int8)
# ... populate boards from your search tree ...

# Parallel legal-move masks for all N boards
masks = ob.batch_get_legal_moves_mask(boards, color=1)

# Parallel score evaluation
diffs = ob.batch_count_diff(boards, color=1)

# Decode a single mask to move list
def mask_to_moves(mask):
    return [(b // 8, b % 8) for b in range(64) if (mask >> b) & 1]

print(mask_to_moves(masks[42]))
```

---

## 12. Design Decisions & Trade-offs

| Decision | Alternative considered | Reason chosen |
|----------|----------------------|---------------|
| Two separate `uint64_t` bitmaps | Single array of 64 bytes | Fewer memory loads; all logic fits in registers |
| 8-call fan-out for directions | Loop with function pointer array | Compiler fully inlines each call; no indirect branch |
| `ray_flips` uses a `while` loop | Unrolled like `legal_dir` | Ray length is data-dependent; unrolling wouldn't help |
| `legal_dir` fully unrolled | `for` loop with `#pragma unroll` | Ensures exactly 6 steps regardless of compiler version |
| Batch functions as free functions | Methods on a "batch" class | Simpler API; no object lifetime concerns |
| `schedule(dynamic, 64)` | `static` | Work per board is non-uniform (depends on board state); dynamic avoids stragglers |
| Stack-allocated `BitBoard` per thread | Thread-local storage | Zero contention; works with any number of threads |
| `alignas(64)` on class | No alignment | Negligible code cost; eliminates false-sharing risk in batch path |
| `#ifdef _OPENMP` guards | Always require OpenMP | Allows use in environments where OpenMP is unavailable (e.g., WASM, some CI) |
