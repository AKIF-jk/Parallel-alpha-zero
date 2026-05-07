# Technique D: Two-Process Parallel Batched Self-Play

## Overview

Technique D adds multiprocessing on top of Technique C. Instead of one process running all batched self-play games, two spawned worker processes each run half of the self-play workload:

```text
Worker 0: BatchedSelfPlayWorker.execute_batch(numEps // 2)
Worker 1: BatchedSelfPlayWorker.execute_batch(numEps // 2)
Main process: collect examples, train network, run arena, save checkpoints
```

The goal was to use the 2 CPU cores available on Google Colab T4 while keeping neural network training and arena comparison in the main process.

## Implementation Details

### Changes Made

- **parallel_coach.py**
  - Added `ParallelCoach`.
  - Added `worker_fn(worker_id, game, nnet, args, result_queue)`.
  - Spawns two `torch.multiprocessing.Process` workers.
  - Collects each worker's examples and stats through `mp.Queue`.
  - Merges examples in the main process.
  - Runs training, arena comparison, model acceptance/rejection, greedy evaluation, and checkpointing in the main process.
  - Saves an iteration checkpoint after every iteration:
    ```text
    iter_001.pth.tar
    iter_002.pth.tar
    ...
    ```

- **run_technique_d.py**
  - Added Technique D entry point.
  - Uses Colab-safe spawn multiprocessing:
    ```python
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    ```
  - Saves metrics to:
    ```text
    CHECKPOINT_DIR/technique_d_metrics.json
    ```

- **batched_selfplay.py**
  - Added worker-local stats so subprocess results can be returned to the main process:
    - `gpu_batch_boards`
    - `gpu_batch_calls`
    - `mcts_sim_count`
  - Added `batch_stats()` for Technique D aggregation.

- **profiler.py**
  - Added:
    ```python
    worker_utilization = []
    examples_per_worker = []
    ```
  - Saves both fields to metrics JSON.

- **utils.py**
  - Fixed `dotdict` so it can be pickled/unpickled by multiprocessing `spawn`.
  - Without this, spawned workers crashed with:
    ```text
    KeyError: '__setstate__'
    ```
  - Updated implementation:
    ```python
    class dotdict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __getstate__(self):
            return dict(self)

        def __setstate__(self, state):
            self.update(state)
    ```

## Final Evaluation Parameters

Technique D used the same stronger evaluation settings as the final Technique C run:

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

Each worker ran:

```python
args.numEps // 2
```

So with `numEps = 48`, each worker generated self-play from `24` games.

## Metrics

```json
{
  "self_play_sec": [54.49, 62.61, 60.50, 63.01, 62.81],
  "train_sec": [36.75, 76.83, 114.11, 143.47, 186.34],
  "arena_sec": [91.89, 84.43, 80.39, 93.15, 103.41],
  "gpu_utilization_pct": [40.30, 48.56, 54.56, 57.39, 60.54],
  "mcts_sims_per_sec": [999.43, 873.75, 903.13, 866.03, 872.01],
  "peak_ram_mb": [1710.45, 2037.96, 2037.96, 2327.13, 2402.82],
  "cache_hit_rate_per_iter": [9.38, 11.07, 6.87, 4.39, 8.22],
  "gpu_calls_per_iter": [3986, 3807, 3740, 3715, 3671],
  "avg_gpu_batch_size": [10.64, 9.15, 9.04, 9.09, 8.82],
  "worker_utilization": [89.73, 88.57, 91.55, 91.67, 91.67],
  "examples_per_worker": [
    [6248, 6200],
    [6248, 6256],
    [6232, 6256],
    [6224, 6248],
    [6248, 6272]
  ],
  "win_rate_vs_greedy": 0.3
}
```

## Verification

| Check | Status | Details |
|-------|--------|---------|
| Uses `spawn`, not `fork` | Pass | `mp.set_start_method('spawn', force=True)` |
| No CUDA fork initialization crash | Pass | Run completed |
| Workers complete | Pass | Metrics saved successfully |
| Checkpoints saved every iteration | Pass | `iter_001.pth.tar` etc. are saved |
| Worker utilization | Pass | Roughly `88-92%` |
| Examples balanced between workers | Pass | Worker counts are within 20% |
| Avg GPU batch size > 1.0 | Pass | `8.82-10.64` |
| Self-play faster than Technique C | Fail | Similar wall-clock, no improvement |
| MCTS throughput better than Technique C | Fail | Technique D much lower |
| Win rate vs greedy within baseline | Fail | `0.3` vs Technique C/Technique A `1.0` |

## Comparison With Technique C

| Metric | Technique C Final | Technique D | Result |
|--------|-------------------|-------------|--------|
| Self-play time | `54-63 sec` | `54-63 sec` | No speedup |
| Avg MCTS sims/sec | ~`1684` | ~`903` | Technique D slower |
| Avg GPU batch size | ~`11.17` | ~`9.35` | Technique D lower |
| GPU calls per iteration | `2841-3286` | `3671-3986` | Technique D more calls |
| Peak RAM iter 5 | `2620.96 MB` | `2402.82 MB` | Technique D lower RAM |
| Win rate vs greedy | `1.0` | `0.3` | Technique D worse |

## Analysis

Technique D is mechanically correct but does not improve performance on Colab T4.

The worker split is balanced:

```text
examples_per_worker ~= equal
worker_utilization ~= 90%
```

So the problem is not load imbalance. The likely causes are:

1. **Single GPU contention**
   - Both worker processes perform CUDA inference.
   - They compete for the same T4 GPU instead of forming one larger global batch.

2. **CUDA multiprocessing overhead**
   - `spawn` avoids Colab's fork/CUDA crash, but it adds process startup and IPC overhead.
   - Sharing CUDA tensors across processes also caused a cleanup warning:
     ```text
     CudaIPCTypes.cpp: Producer process has been terminated before all shared CUDA tensors released
     ```

3. **Duplicated search state**
   - Technique C has one shared batched self-play worker with one shared tree/cache.
   - Technique D has one worker-local tree/cache per process.
   - This loses some transposition and cache reuse across the full set of 48 games.

4. **Smaller effective batches per process**
   - Technique C batches across the full active self-play set.
   - Technique D splits the games into two groups of 24.
   - Each worker's batch is smaller and less stable than a single global coordinator.

5. **Quality regression**
   - Final win rate vs greedy dropped to `0.3`.
   - This suggests the generated self-play data was lower quality or more unstable than Technique C's data.

## Improvements

Technique D still produced some useful results:

1. **Multiprocessing was made Colab-safe**
   - `spawn` mode worked.
   - The initial `dotdict` pickling crash was fixed.

2. **Workers were balanced**
   - Example counts were nearly identical across workers.
   - This means the static split of `numEps // 2` was acceptable for workload balance in this run.

3. **Worker utilization was high**
   - Worker utilization stayed near `90%`.
   - The workers were active; the bottleneck was not worker idleness.

4. **Checkpoints were saved every iteration**
   - This protects against Colab session timeouts.

## Drawbacks

1. **No self-play speedup**
   - Self-play time did not improve over Technique C.

2. **Lower throughput**
   - MCTS simulations per second dropped from Technique C's ~`1684` to Technique D's ~`903`.

3. **More GPU calls**
   - Technique D made more batched GPU calls than Technique C.

4. **Lower average GPU batch size**
   - Technique D's average batch size was lower than Technique C's final run.

5. **Poor win rate**
   - `win_rate_vs_greedy = 0.3`, which fails the quality check.

6. **More complex and fragile**
   - Requires spawn-safe objects.
   - Requires queue communication.
   - Can emit CUDA IPC cleanup warnings.

## Recommendation

Technique D should be reported as a negative multiprocessing result.

The better architecture would be:

```text
Worker processes:
  CPU-side MCTS traversal only
  send pending leaf boards to main process

Main process:
  owns the CUDA model
  batches all pending boards from all workers
  runs one global GPU inference queue
  sends policy/value results back to workers
```

That design avoids multiple CUDA contexts and preserves larger global batches. It is more complex, but it is the correct direction if multiprocessing must be continued.

For this project, Technique C remains the best implementation because it achieves high batch size, high MCTS throughput, and baseline-quality win rate without multiprocessing overhead.

## Conclusion

Technique D successfully demonstrates two-process parallel self-play with Colab-safe `spawn`, balanced worker output, and iteration checkpointing. However, it fails the main performance and quality goals. It does not beat Technique C in self-play speed, reduces MCTS throughput, increases GPU call count, and produces a weaker model.

Final verdict: functional but not recommended. Technique C should remain the primary optimization.
