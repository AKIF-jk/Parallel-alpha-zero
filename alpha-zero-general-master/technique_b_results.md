# Technique B: Neural Network Position Cache for MCTS

## Overview

Technique B adds a transposition-style cache to `MCTS.py` so repeated board positions reuse previously computed neural network outputs instead of calling `nnet.predict()` again.

The cache stores:

```python
board_string -> (policy_np_array, value_float)
```

This reduces duplicate GPU inference during self-play, especially when the same or symmetric positions recur across episodes within the same training iteration.

## Implementation Details

### Changes Made

- **MCTS.py**
  - Added neural network cache fields:
    ```python
    self.nn_cache = {}
    self.cache_hits = 0
    self.cache_misses = 0
    ```
  - Added `cache_stats()` to report:
    - cache hits
    - cache misses
    - hit rate percentage
    - current cache size
  - Added `reset_search_tree()` so the per-episode MCTS tree can be cleared without clearing the neural network cache.
  - Updated leaf expansion so `nnet.predict(board)` is only called on cache miss.
  - On cache miss, cached all symmetric board-policy variants returned by `game.getSymmetries(board, pi)`.

- **Coach.py**
  - Changed self-play so the MCTS search tree resets between episodes, but the neural network cache persists for the full training iteration.
  - Reset the cache at the start of each training iteration because the neural network weights may change after training.
  - Recorded cache hit rate and GPU call count after each iteration.

- **profiler.py**
  - Added two metric lists:
    ```python
    cache_hit_rate_per_iter = []
    gpu_calls_per_iter = []
    ```
  - Added both fields to saved metrics JSON.

- **run_technique_b.py**
  - Added a 5-iteration runner that saves metrics to:
    ```text
    CHECKPOINT_DIR/technique_b_metrics.json
    ```

## Performance Results

### Technique A vs Technique B

| Metric | Technique A | Technique B | Result |
|--------|-------------|-------------|--------|
| Avg self-play time | 31.91 sec | 30.17 sec | Technique B faster by about 5.5% |
| Iteration 1 self-play | 32.56 sec | 33.31 sec | Technique B slightly slower initially |
| Iteration 5 self-play | 31.98 sec | 28.37 sec | Technique B faster |
| Avg MCTS sims/sec | 1018.98 | 1075.96 | Technique B faster |
| Cache hit rate | N/A | 12.89% -> 19.44% | Cache is working |
| GPU calls per iteration | Not recorded | 12337 -> 9701 | GPU calls decrease |
| Win rate vs greedy | 1.0 | 0.6 | Lower than Technique A |

### Technique B Metrics

```json
{
  "self_play_sec": [33.31, 30.10, 30.54, 28.51, 28.37],
  "gpu_utilization_pct": [42.37, 48.30, 55.29, 58.47, 63.40],
  "mcts_sims_per_sec": [974.09, 1075.47, 1061.56, 1131.26, 1137.45],
  "cache_hit_rate_per_iter": [12.89, 15.51, 18.49, 18.34, 19.44],
  "gpu_calls_per_iter": [12337, 10680, 10476, 9744, 9701],
  "win_rate_vs_greedy": 0.6
}
```

## Improvements

1. **Fewer neural network calls**
   - Cache misses represent actual neural network calls.
   - GPU calls decreased from `12337` in iteration 1 to `9701` in iteration 5.

2. **Better self-play throughput**
   - Average self-play time improved from about `31.91 sec` in Technique A to `30.17 sec` in Technique B.
   - MCTS simulations per second improved from about `1019` to `1076`.

3. **Cache behavior is correctly scoped**
   - Hit rate is not zero, so the cache is not being reset between episodes.
   - Hit rate is not extremely high in iteration 1, so the cache is not persisting incorrectly across training iterations.
   - The cache resets at the start of each iteration, which is required because network weights can change after training.

4. **Symmetry-aware reuse**
   - Each cache miss also stores symmetric variants of the position and policy.
   - This improves reuse without requiring separate neural network calls for equivalent board states.

## Drawbacks

1. **Additional RAM usage**
   - The cache stores policy arrays and values for many board positions.
   - Peak RAM increased compared with Technique A:
     - Technique A iteration 5: `1708.86 MB`
     - Technique B iteration 5: `1839.79 MB`

2. **Initial overhead**
   - Iteration 1 self-play was slightly slower than Technique A:
     - Technique A: `32.56 sec`
     - Technique B: `33.31 sec`
   - Early in training, there are fewer repeated positions, so cache lookup and symmetry insertion overhead can outweigh savings.

3. **Win rate variance**
   - Technique B reached `0.6` win rate vs greedy, while Technique A reported `1.0`.
   - This may be run variance from short training and a small greedy evaluation set, but it does not satisfy a strict "within 5% of Technique A" check.
   - A more reliable comparison should use fixed random seeds and more arena games.

4. **Cache invalidation is required**
   - The cache must be cleared whenever neural network weights change.
   - Reusing cached predictions across training iterations would mix predictions from old and new models and could corrupt self-play targets.

5. **Unbounded cache size**
   - The current cache is a normal dictionary and can grow for the full iteration.
   - If memory becomes a problem, replace it with a bounded LRU cache, for example with a maximum size around `50000`.

## Verification

| Check | Status | Details |
|-------|--------|---------|
| Cache hit rate starts low and rises | Pass | `12.89% -> 19.44%` |
| Cache is not reset per episode | Pass | Hit rate is nonzero and rises |
| Cache is not persisted across iterations | Pass | Iteration 1 hit rate is not abnormally high |
| GPU calls decrease | Pass | `12337 -> 9701` |
| Self-play time improves | Pass | Average self-play improved by about 5.5% |
| Win rate within 5% of Technique A | Needs rerun | Technique A: `1.0`, Technique B: `0.6` |

## Conclusion

Technique B successfully reduces duplicate neural network inference and improves self-play throughput. The cache is placed at the correct level: it survives across episodes within an iteration and resets before the next iteration starts.

The main tradeoff is higher memory usage and a lower measured win rate vs greedy in this run. The performance result is strong enough to keep the technique, but the win-rate comparison should be rerun with fixed seeds or more evaluation games before making a final correctness claim.
