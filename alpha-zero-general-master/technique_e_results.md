# Technique E: Batched Self-Play With Explicit Virtual Loss

## Overview

Technique E keeps the Technique C single-process batched self-play architecture and makes virtual loss explicit in the MCTS selection path. The goal was to reduce duplicate branch selection among active batched games while preserving the strong GPU batching and quality behavior from Technique C.

This run used the same GPU-backed evaluation settings as Technique C:

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

## Metrics

```json
{
  "self_play_sec": [57.80, 64.04, 64.73, 67.85, 66.53],
  "train_sec": [42.88, 74.78, 112.15, 148.33, 186.10],
  "arena_sec": [88.53, 82.55, 90.37, 97.25, 92.10],
  "gpu_utilization_pct": [37.95, 45.82, 51.21, 55.14, 59.31],
  "mcts_sims_per_sec": [1728.92, 1570.25, 1541.02, 1480.95, 1506.06],
  "peak_ram_mb": [2049.80, 2343.23, 2343.23, 2647.23, 2662.41],
  "cache_hit_rate_per_iter": [11.07, 12.87, 14.56, 19.00, 17.74],
  "gpu_calls_per_iter": [3124, 3006, 2932, 2909, 2831],
  "avg_gpu_batch_size": [13.30, 11.49, 10.83, 10.25, 10.38],
  "avg_virtual_loss_collisions_avoided": [0.048, 0.073, 0.073, 0.077, 0.067],
  "worker_utilization": [],
  "examples_per_worker": [],
  "win_rate_vs_greedy": 1.0
}
```

## Summary Statistics

| Metric | Average | Range |
|--------|---------|-------|
| Self-play time | `64.19 sec` | `57.80-67.85 sec` |
| Training time | `112.85 sec` | `42.88-186.10 sec` |
| Arena time | `90.16 sec` | `82.55-97.25 sec` |
| GPU utilization | `49.89%` | `37.95-59.31%` |
| MCTS sims/sec | `1565.44` | `1480.95-1728.92` |
| Peak RAM | `2409.18 MB` | `2049.80-2662.41 MB` |
| Cache hit rate | `15.05%` | `11.07-19.00%` |
| GPU calls per iter | `2960.40` | `2831-3124` |
| Avg GPU batch size | `11.25` | `10.25-13.30` |
| Avg virtual-loss diversions | `0.067` | `0.048-0.077` |

Average total iteration time was `267.20 sec`, increasing from `189.21 sec` in iteration 1 to `344.73 sec` in iteration 5. The growth is mostly from accumulated training examples, not from self-play.

## Comparison With Technique C

| Metric | Technique C Final | Technique E | Result |
|--------|-------------------|-------------|--------|
| Avg self-play time | ~`59.83 sec` | `64.19 sec` | Technique E slower |
| Avg MCTS sims/sec | ~`1683.67` | `1565.44` | Technique E lower |
| Avg GPU utilization | ~`50.61%` | `49.89%` | Similar |
| Avg GPU batch size | ~`11.17` | `11.25` | Similar/slightly higher |
| Avg GPU calls per iter | ~`2980.40` | `2960.40` | Similar/slightly lower |
| Avg cache hit rate | ~`6.95%` | `15.05%` | Technique E higher |
| Peak RAM iter 5 | `2620.96 MB` | `2662.41 MB` | Technique E slightly higher |
| Win rate vs greedy | `1.0` | `1.0` | Quality preserved |

## Findings

1. **Technique E preserves model quality**
   - `win_rate_vs_greedy = 1.0`.
   - The explicit virtual-loss change did not harm the final greedy comparison.

2. **GPU batching stayed healthy**
   - Average GPU batch size was `11.25`, essentially the same as Technique C.
   - GPU calls stayed around `2.8k-3.1k` per iteration.
   - GPU utilization was also similar to Technique C at roughly `50%`.

3. **Cache hit rate improved**
   - Cache hit rate averaged `15.05%`, higher than Technique C's final run.
   - This suggests Technique E reused more evaluated positions, possibly because explicit virtual-loss behavior changed branch scheduling and duplicate leaf handling.

4. **Self-play did not speed up**
   - Average self-play time increased to `64.19 sec`.
   - Average MCTS throughput dropped to `1565.44` sims/sec.
   - The slowdown indicates that Technique E's added virtual-loss bookkeeping costs more CPU time than it saves through collision avoidance.

5. **Virtual-loss collision avoidance was small**
   - Average virtual-loss diversions were only `0.067` per MCTS simulation.
   - That is too low to justify heavy extra selection-path work.
   - The metric shows the feature is active, but it is not preventing enough duplicated work to create a measurable speedup.

6. **The remaining bottleneck is CPU-side MCTS/game logic**
   - GPU batch size, GPU call count, and GPU utilization were already close to Technique C.
   - The regression appears in `mcts_sims_per_sec`, not GPU batching.
   - This points to CPU traversal, UCB calculation, virtual-loss accounting, game transitions, and dictionary/cache lookups as the limiting path.

## Drawbacks

1. **Lower MCTS throughput than Technique C**
   - Technique E averaged about `7%` lower MCTS simulations per second than Technique C.

2. **Higher self-play wall-clock time**
   - Self-play averaged about `4.36 sec` slower per iteration than Technique C.

3. **Extra hot-path bookkeeping**
   - Selection now compares virtual-loss-adjusted and non-virtual-loss actions to count diversions.
   - That can add duplicate UCB computation in the hottest part of MCTS.

4. **No multiprocessing benefit**
   - `worker_utilization` and `examples_per_worker` are empty because Technique E is still single-process batched self-play.
   - CPU-side MCTS remains constrained to one Python process.

## Recommendation

Technique E should be reported as a quality-preserving but speed-negative experiment.

Keep the correctness parts of explicit virtual loss only if they are needed for batched search behavior, but remove or gate the expensive metrics from the hot path:

- avoid computing both virtual-loss and non-virtual-loss UCB every selection unless profiling is enabled
- keep diversion tracking behind a debug/profile flag
- cache repeated node fields locally inside `_select_action`
- profile `getValidMoves`, `getNextState`, `getGameEnded`, `stringRepresentation`, `_select_action`, `_complete_leaf`, and backup time separately

For further speedup, focus on CPU parallelism rather than more GPU batching:

```text
Worker processes:
  CPU-side MCTS traversal and Othello game logic
  process-local trees and caches
  send leaf boards to the parent

Main process:
  owns the CUDA model
  coalesces duplicate boards
  runs one global batched inference call
  returns policy/value results to workers
```

This is the most promising direction because Technique E shows the GPU path is already batched well, while CPU-side MCTS throughput is the part that regressed.

## Conclusion

Technique E successfully preserves playing strength and keeps GPU batching effective, but it does not improve runtime. The higher cache hit rate and stable batch size are useful signals, but they are outweighed by lower MCTS throughput and slower self-play.

Final verdict: functional and quality-safe, but not a performance win. Technique C remains the better single-process result unless Technique E's virtual-loss bookkeeping is optimized or moved behind profiling flags.
