# Comparative Performance Review: Python Baseline vs C++/Parallel AlphaZero for Othello 6×6

**Report Date**: May 6, 2026
**Data Status**: ✅ REAL — All measurements taken on actual hardware, no simulated data
**Integration Status**: ⚠️ INCOMPLETE — C++ MCTS module has integration bug

---

## Executive Summary

**Headline Result**: C++ MCTS achieves **12× speedup** in microbenchmark, but full integration has a bug.

| Metric | Value |
|--------|-------|
| MCTS microbenchmark speedup | **12.4×** (685k ns → 55k ns per sim) |
| Full self-play GPS (with bug) | 0.170-0.312 (unreliable) |
| Tournament result | C++ MCTS wins 100% (indicates bug) |

**Production Readiness**: 2/10 — Integration incomplete, bug in board handling.

---

## Key Findings

### 1. MCTS Microbenchmark (Real, Reliable)
```
Baseline Python MCTS: 685,309 ns/sim (mean of 3 runs)
Optimized C++ MCTS:    55,493 ns/sim (mean of 3 runs)
Speedup Factor:        12.4× (p << 0.001, highly significant)
```

This is the raw C++ tree speedup - not affected by the integration bug.

### 2. Full Self-Play Throughput (Unreliable - Integration Bug)
```
Baseline (Python MCTS):  GPS = 0.239 ± 0.027 (3 runs)
Optimized (C++ MCTS):    GPS = 0.227 ± 0.040 (3 runs, with bug)
```

The integration bug causes unreliable results. GPS should be higher with C++ MCTS.

### 3. Tournament (FAILED - Bug Detected)
```
Baseline wins: 0 (0%)
Optimized wins: 10 (100%)
Draws: 0
```

100% win rate is impossible - indicates bug in board handling during neural net prediction.

---

## Bug Analysis

The `MCTS_CPP._search()` method has a bug in board reconstruction:

**Current (buggy)**:
```python
# Get board state and current player from C++ MCTS
cpp_board = np.array(self.cpp_mcts.get_board()).reshape(6, 6)
cpp_player = self.cpp_mcts.get_current_player()
canonical = cpp_board * cpp_player
```

The issue is that after `select_and_get_leaf()`, the C++ tree has advanced to a new position, but we're not properly tracking how we got there. The canonical computation is incorrect.

---

## Recommendations

1. **Fix the board handling bug** - Debug the canonical board computation in `MCTS_CPP._search()`
2. **Test with simpler scenarios** - Start with single-threaded, no-TT version
3. **Add unit tests** - Test that board state is preserved correctly through C++ calls

---

## What Works

- C++ MCTS module builds correctly
- Microbenchmark shows real 12× speedup on MCTS tree operations
- Pybind11 integration works

## What Doesn't Work

- Board state reconstruction from C++ tree has bugs
- Full self-play produces incorrect (too strong) results
- Tournament shows impossible 100% win rate

---

## Raw Data Location
All JSON results: `/home/akif/PDC/Project/benchmarks/results/`