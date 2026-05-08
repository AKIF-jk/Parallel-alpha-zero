# Optimization Integration Log (Phases 1-3)

This document summarizes the integrated changes from `revised_plan/optimization_plan.md`, what was benchmarked, and the current recommended settings.

## Scope

- Target repo: `alpha-zero-general-master`
- Roadmap applied phase-wise:
  - Phase 1: Quick wins (hashing, cache bounds, pooling, profiling)
  - Phase 2: Parallel MCTS (thread-pool approach)
  - Phase 3: C++ bitboard backend integration (optional)

## Phase 1 Changes

### Added utilities
- `optimization_utils.py`
  - `LRUCache` (bounded cache)
  - `NumpyArrayPool` (reusable fixed-shape arrays)
  - `ZobristHash` (state hashing helper)

### Integrated in search/self-play
- `MCTS.py`
  - Replaced unbounded NN cache with `LRUCache`
  - Added pooled action-count arrays in `getActionProb`
  - Added cache-safe lookup/insert paths
- `batched_selfplay.py`
  - Replaced `nn_cache` and `terminal_cache` with bounded `LRUCache`
  - Added pooled action-count arrays
- `parallel_coach.py`
  - Replaced inference cache with bounded `LRUCache`
- `Coach.py`
  - Cache reset now uses `clear()` on cache object

### Integrated in game representation
- `othello/OthelloGame.py`
  - Optional Zobrist-backed `stringRepresentation`
  - Reduced unnecessary board copying in some paths

### Added profiling helper
- `profile_phase1.py`
  - Deterministic MCTS micro-benchmark with cProfile output

### Config flags added to run scripts
Added defaults to:
- `main.py`
- `run_baseline.py`
- `run_technique_a.py`
- `run_technique_b.py`
- `run_technique_c.py`
- `run_technique_d.py`
- `run_technique_e.py`

New knobs:
- `nnCacheMaxSize`
- `terminalCacheMaxSize`
- `inferenceCacheMaxSize`
- `actionArrayPoolSize`

## Phase 2 Changes

### Parallel MCTS integration
- `MCTS.py`
  - Added optional thread-pool simulation mode
  - Added node-level locks and per-action locks
  - Added virtual-loss coordination for concurrent traversal
  - Added configurable `numMCTSThreads`
  - Added persistent executor reuse to reduce pool setup overhead

### Config updates
- Added `numMCTSThreads` to all main run configs (default `1`).

## Phase 3 Changes

### C++ bitboard backend (optional)
- Added `othello/othello_bitboard.cpp` (pybind11 module `othello_bitboard`)
  - 8x8 bitboard representation
  - Legal move generation
  - Move execution/flips
  - Score diff
  - Board import/export to numpy
  - Native hash

### Python integration
- `othello/OthelloGame.py`
  - Added optional `use_bitboard=True` backend path (8x8 only)
  - Automatic fallback to Python logic if extension unavailable

### Benchmark script
- `benchmark_phase3.py`
  - Compares:
    - `python_tobytes`
    - `python_zobrist`
    - `bitboard_cpp`
  - Reports `SIMS/SEC`, elapsed time, relative speed, and cache stats

## Benchmarks and Outcome

From your latest benchmark:

```
python_tobytes: 1220.5 sims/sec (REL 1.00)
python_zobrist:  835.4 sims/sec (REL 0.68)
bitboard_cpp:   1104.2 sims/sec (REL 0.90)
```

Interpretation:
- Current best path is `python_tobytes`.
- Current Zobrist implementation is slower for this workload.
- C++ bitboard is faster than Zobrist but still behind baseline due to Python<->C++ conversion overhead in hot loops.

## Recommended Runtime Settings (Current)

Use these until further optimization:

- `use_zobrist=False`
- `numMCTSThreads=1`
- `use_bitboard=False` for production runs

Use `benchmark_phase3.py` to re-evaluate after further changes.

## Known Gaps / Next Work

To make Phase 3 outperform baseline consistently:

1. Keep board state in native bitboard form through MCTS traversal.
2. Minimize numpy conversions (convert mainly at NN inference boundaries).
3. Reduce lock/contention overhead if parallel search is revisited.

This is effectively a Phase 3b refactor.
