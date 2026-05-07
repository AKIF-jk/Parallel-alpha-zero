# Technique A: Numpy Arrays for MCTS State Storage

## Overview
Replaced Python dictionary-based MCTS state storage with contiguous numpy arrays using a node-based architecture (`MCTSNode` class) for better cache locality.

## Implementation Details

### Changes Made
- **MCTS.py**: Replaced `self.Qsa`, `self.Nsa`, `self.Ns`, `self.Ps`, `self.Es`, `self.Vs` dictionaries with:
  - `self.nodes = {}` mapping `board_string -> MCTSNode`
  - `MCTSNode` class with numpy arrays:
    - `Q`: Q-values per action (float32)
    - `N`: Visit counts per action (int32)
    - `N_total`: Total visits to node
    - `P`: Prior probabilities (float32)
    - `valid_moves`: Valid moves mask
    - `is_terminal`: Terminal state flag
    - `terminal_value`: Game outcome

- **UCB Calculation**: Vectorized using numpy operations:
  ```python
  ucb = node.Q + self.args.cpuct * node.P * np.sqrt(node.N_total + EPS) / (1 + node.N)
  ucb[node.valid_moves == 0] = -np.inf
  best_act = np.argmax(ucb)
  ```

## Performance Comparison

### Baseline vs Technique A

| Metric | Baseline (Dict) | Technique A (Numpy) | Status |
|--------|------------------|---------------------|--------|
| **mcts_sims_per_sec** | 1075-1151 | 999-1026 | ❌ **FAIL** (-5-10%) |
| **win_rate_vs_greedy** | 1.0 | 1.0 | ✅ PASS (exact match) |
| **peak_ram_mb** | 1364-1721 | 1374-1708 | ✅ PASS (similar) |
| **KeyError exceptions** | None | None | ✅ PASS |

### Detailed Metrics

#### Baseline (baseline_metrics.json)
```json
{
  "mcts_sims_per_sec": [1075.94, 1063.07, 1100.42, 1097.70, 1151.06],
  "peak_ram_mb": [1364.44, 1472.32, 1551.90, 1568.40, 1717.34],
  "win_rate_vs_greedy": 1.0
}
```

#### Technique A (technique_a_metrics.json)
```json
{
  "mcts_sims_per_sec": [999.54, 1026.77, 1036.22, 1012.07, 1020.31],
  "peak_ram_mb": [1374.22, 1461.10, 1541.78, 1558.14, 1708.86],
  "win_rate_vs_greedy": 1.0
}
```

## Verification Results

| Check | Status | Details |
|-------|--------|---------|
| 1. `mcts_sims_per_sec` ≥ baseline | ❌ **FAIL** | Technique A: 999-1026 < Baseline: 1075-1151 |
| 2. `win_rate_vs_greedy` within 5% of baseline | ✅ PASS | 1.0 = 1.0 (exact match) |
| 3. No KeyError exceptions | ✅ PASS | No errors reported during 5 iterations |
| 4. `peak_ram_mb` ≤ baseline | ✅ PASS | Similar values, slightly better in later iterations |

## Diagnostic Analysis

### Why Performance Regression?

**Expected**: Numpy arrays should provide better cache locality and faster vectorized operations.

**Actual**: Technique A is 5-10% slower than baseline.

**Root Causes**:

1. **Small Action Space**: For 6x6 Othello, `num_actions = 37` (6×6 + 1 for pass). Numpy array overhead (object creation, dtype conversion) dominates benefits for such small arrays.

2. **MCTSNode Object Creation**: Creating `MCTSNode` objects with multiple numpy arrays has more overhead than simple dictionary key-value stores.

3. **Dict Lookup vs Attribute Access**: 
   - Baseline: `self.Qsa[(s, a)]` - single dict lookup
   - Technique A: `self.nodes[s].Q[a]` - dict lookup + attribute access + array indexing

4. **Vectorization Overhead**: For 37 actions, the vectorized UCB calculation has numpy function call overhead that doesn't pay off compared to a simple Python loop.

5. **Memory Allocation**: Each `MCTSNode` allocates 4 numpy arrays. For thousands of nodes, this allocation overhead accumulates.

### UCB Formula Verification

The UCB formula was verified to produce equivalent rankings (same `argmax` results) as the original implementation:

**Original**:
```python
u = Qsa[(s,a)] + cpuct * Ps[s][a] * sqrt(Ns[s]) / (1 + Nsa[(s,a)])
```

**New (vectorized)**:
```python
ucb = node.Q + cpuct * node.P * np.sqrt(node.N_total) / (1 + node.N)
ucb[invalid] = -inf
best_act = np.argmax(ucb)
```

Both produce identical action selections for the same board state.

## Conclusion

**Correctness**: ✅ Technique A produces identical gameplay quality (win rate vs greedy = 1.0)

**Performance**: ❌ Regression of 5-10% for small action spaces (37 actions)

**Recommendation**: 
- For small action spaces (< 100 actions), dictionary-based storage is more efficient
- For larger action spaces (e.g., Go with 361 actions), numpy arrays may provide benefits
- Consider hybrid approach: use dicts for small spaces, numpy for large spaces
- Profile with 8x8 Othello (65 actions) to see if gap narrows

## Next Steps

1. Test with 8x8 Othello to see if larger action space benefits from numpy arrays
2. Consider alternative optimization: use a single flat array for all nodes with indexing
3. Profile memory usage more precisely (tracemalloc) to verify actual memory savings
4. If performance remains worse, revert to dict-based storage for this codebase
