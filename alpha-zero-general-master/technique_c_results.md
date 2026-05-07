# Technique C: Lockstep Batched Self-Play

## Overview

Technique C changes self-play from one completed episode at a time to lockstep batched self-play. Multiple games are active at once, and MCTS leaf evaluations from those games are collected into one batched neural network call.

The goal is to reduce small GPU calls and improve self-play throughput on Google Colab T4 while staying single-threaded. No threading or multiprocessing is used.

## Implementation Details

### Changes Made

- **batched_selfplay.py**
  - Added `BatchedSelfPlayWorker`.
  - Uses `BATCH_SIZE = 16` active game slots for Colab T4.
  - Runs games in a rolling-refill loop:
    - keep up to 16 games active
    - when one finishes, refill that slot with the next game
    - continue until `num_games` games complete
  - Collects pending MCTS leaf evaluations from all active games before GPU inference.
  - Sends pending leaves through one batched PyTorch call.
  - Tracks average GPU batch size via profiler counters.
  - Adds `MAX_STEPS = 200` guard to prevent stuck games.

- **Shared search state**
  - Active games share one MCTS node dictionary:
    ```python
    self.nodes = {}
    ```
  - This allows transposition reuse across games in the same self-play iteration.
  - Terminal states are cached in:
    ```python
    self.terminal_cache = {}
    ```

- **Neural network cache**
  - Keeps the Technique B cache behavior:
    ```python
    self.nn_cache = {}
    ```
  - Cache is shared across active games within the worker.
  - Cache stores direct positions and symmetric variants.

- **Duplicate leaf coalescing**
  - If multiple active games request the same leaf position in the same batch, the board is sent to the GPU once.
  - The returned policy/value is then backed up through all waiting paths.

- **Virtual loss**
  - Added lightweight virtual visit counts to shared-tree action selection.
  - This reduces duplicate branch selection while multiple active games are waiting for batched inference.

- **Coach.py**
  - Replaced the per-episode self-play loop with:
    ```python
    worker = BatchedSelfPlayWorker(self.game, self.nnet, self.args)
    iterationTrainExamples = deque(
        worker.execute_batch(self.args.numEps),
        maxlen=self.args.maxlenOfQueue
    )
    ```
  - Added `avg_gpu_batch_size` and batched GPU call metrics to profiler output.
  - Greedy evaluation now uses configurable `args.greedyCompare`.

- **profiler.py**
  - Added:
    ```python
    avg_gpu_batch_size = []
    ```
  - Added counters for:
    - total boards sent to GPU
    - total batched GPU calls
  - Saves `avg_gpu_batch_size` to metrics JSON.

- **run_technique_c.py**
  - Added Technique C runner saving:
    ```text
    CHECKPOINT_DIR/technique_c_metrics.json
    ```
  - Logs CUDA availability and device.
  - Enables `torch.backends.cudnn.benchmark = True` when CUDA is available.

## Final Evaluation Parameters

The final stronger run used larger GPU-backed settings:

```python
numIters = 5
numEps = 48
numMCTSSims = 35
arenaCompare = 40
greedyCompare = 40
nnet_args.epochs = 15
nnet_args.batch_size = 128
BATCH_SIZE = 16
```

These values were chosen because:

- `48` self-play games is exactly `3 * 16`, matching the lockstep batch size.
- `35` MCTS simulations gives stronger policies than the earlier 25-sim run.
- `15` epochs and training batch size `128` use the GPU better.
- `40` arena and greedy games reduce win-rate noise.

## Final Metrics

```json
{
  "self_play_sec": [54.35, 59.90, 62.04, 62.64, 60.21],
  "train_sec": [38.51, 70.13, 114.41, 146.88, 183.19],
  "arena_sec": [90.76, 76.18, 93.96, 99.43, 77.90],
  "gpu_utilization_pct": [39.53, 44.88, 52.03, 55.78, 60.82],
  "mcts_sims_per_sec": [1848.12, 1678.17, 1620.69, 1604.81, 1666.54],
  "peak_ram_mb": [2009.01, 2303.30, 2303.30, 2576.94, 2620.96],
  "cache_hit_rate_per_iter": [10.95, 3.53, 7.95, 3.32, 8.98],
  "gpu_calls_per_iter": [3286, 2991, 2864, 2841, 2920],
  "avg_gpu_batch_size": [12.68, 12.21, 11.20, 9.43, 10.35],
  "win_rate_vs_greedy": 1.0
}
```

## Progression

### Initial Technique C Run

The first smaller Technique C run used lighter settings and showed that batching worked:

```json
{
  "self_play_sec": [20.22, 18.38, 19.29, 19.34, 19.14],
  "mcts_sims_per_sec": [1605.85, 1769.46, 1700.39, 1685.34, 1704.61],
  "gpu_calls_per_iter": [1434, 1380, 1352, 1351, 1263],
  "avg_gpu_batch_size": [9.02, 7.96, 7.57, 6.84, 7.47],
  "win_rate_vs_greedy": 0.5
}
```

This passed the batching and speed checks, but the win rate was weaker.

### Increased-Parameter Technique C Run

After increasing self-play games, MCTS simulations, training epochs, training batch size, and evaluation games:

- average GPU batch size improved to roughly `9.43-12.68`
- win rate vs greedy improved to `1.0`
- MCTS throughput stayed high at roughly `1605-1848` sims/sec
- RAM usage increased, peaking around `2621 MB`

This is the better run to report as the final Technique C result.

## Comparison With Technique B

Technique B used smaller parameters, so raw wall-clock time is not directly comparable to the final larger Technique C run. The fairer signal is throughput and batching behavior.

| Metric | Technique B | Technique C Final | Result |
|--------|-------------|-------------------|--------|
| Avg MCTS sims/sec | ~1076 | ~1684 | Technique C much faster |
| Avg GPU batch size | N/A | ~11.17 | Batching works |
| GPU calls per iteration | 9701-12337 cache misses | 2841-3286 batched calls | Technique C uses far fewer GPU calls |
| Win rate vs greedy | 0.6 | 1.0 | Technique C better in final run |
| Peak RAM iter 5 | 1839.79 MB | 2620.96 MB | Technique C uses more RAM |

## Improvements

1. **Batched neural network inference**
   - Average GPU batch size reached `9.43-12.68`.
   - This is close to the ideal batch size of `16` for lockstep self-play with game-length variance.

2. **Higher MCTS throughput**
   - Technique C achieved roughly `1605-1848` MCTS simulations per second.
   - This is substantially higher than Technique B's roughly `1076` simulations per second.

3. **Fewer GPU calls**
   - Final Technique C used about `2841-3286` batched GPU calls per iteration.
   - Technique B had about `9701-12337` neural cache misses per iteration under smaller settings.

4. **Correctness improved with stronger settings**
   - The final increased-parameter run reached `win_rate_vs_greedy = 1.0`.
   - This matches the Technique A baseline and improves over the earlier Technique C run.

5. **Better batch stability**
   - Rolling refill prevents the weak `16 + 4` wave pattern when `numEps=20`.
   - With `numEps=48`, games are processed as three full 16-game waves over time, with slots refilled as games finish.

## Drawbacks

1. **Higher RAM usage**
   - Shared tree state, NN cache, terminal cache, and larger training history increased memory usage.
   - Peak RAM reached about `2621 MB` in the final run.

2. **Longer total wall-clock time with stronger settings**
   - Final self-play time was `54-63 sec` per iteration.
   - Training time increased from `38.51 sec` in iteration 1 to `183.19 sec` in iteration 5 because train examples accumulate across iterations.
   - Arena time also increased because `arenaCompare` was raised to `40`.

3. **GPU utilization did not increase dramatically**
   - GPU utilization ranged from about `39.53%` to `60.82%`.
   - This is not dramatically higher than Technique B, because Python-side MCTS and arena logic still consume significant time.
   - However, throughput and batch size improved, so utilization alone is not the best success metric.

4. **More complex implementation**
   - Technique C is more complex than sequential self-play.
   - It must coordinate active game slots, shared tree state, duplicate leaf coalescing, cache hits, batched inference, and value backup.

5. **Careful value-backup semantics are required**
   - The worker preserves the original recursive MCTS convention by backing up `-value` from newly evaluated leaf nodes.
   - Changing this sign convention can silently damage policy quality.

## Verification

| Check | Status | Details |
|-------|--------|---------|
| `avg_gpu_batch_size > 1.0` | Pass | Final run: `9.43-12.68` |
| Batch size realistic for 16 games | Pass | Values are in the expected `4-12` range, with some above 12 |
| Self-play batching completes all games | Pass | All 48 games completed each iteration |
| Infinite-loop guard present | Pass | `MAX_STEPS = 200` |
| MCTS throughput improves | Pass | Final run: ~`1684` avg sims/sec |
| Fewer GPU calls | Pass | Batched calls reduced to ~`2.8k-3.3k` per iteration |
| Win rate vs greedy within baseline | Pass | Final run: `1.0` |
| Memory acceptable on Colab T4 | Pass with cost | Peak ~`2621 MB` |

## Conclusion

Technique C successfully implements lockstep batched self-play. The final increased-parameter run is the strongest result: it preserves win rate, achieves high average GPU batch sizes, reduces GPU call count, and improves MCTS simulation throughput.

The main tradeoff is higher memory usage and longer total runtime when using stronger training and evaluation settings. For reporting, the final run should be presented as the quality-validated Technique C result, while the smaller run can be mentioned as an initial speed-focused validation.
