# Technique D: Two-Process Parallel Batched Self-Play

## Overview

Technique D adds multiprocessing on top of Technique C. Instead of one process running all batched self-play games, two spawned worker processes each run half of the self-play workload:

```text
Worker 0: BatchedSelfPlayWorker.execute_batch(numEps // 2)
Worker 1: BatchedSelfPlayWorker.execute_batch(numEps // 2)
Main process: collect examples, train network, run arena, save checkpoints
```

The goal was to use the 2 CPU cores available on Google Colab T4 while keeping neural network training and arena comparison in the main process.

The first implementation let each worker own a CUDA inference path. A later revision changed the architecture so workers do CPU-side MCTS traversal only and the main process owns all PyTorch inference.

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

In the historical two-CUDA-worker run, each worker ran:

```python
args.numEps // 2
```

So with `numEps = 48`, each worker generated self-play from `24` games.

## Historical Metrics: Two CUDA Workers

These metrics are from the first Technique D implementation, where both workers performed CUDA inference independently.

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

## Historical Verification

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

## Historical Comparison With Technique C

| Metric | Technique C Final | Technique D | Result |
|--------|-------------------|-------------|--------|
| Self-play time | `54-63 sec` | `54-63 sec` | No speedup |
| Avg MCTS sims/sec | ~`1684` | ~`903` | Technique D slower |
| Avg GPU batch size | ~`11.17` | ~`9.35` | Technique D lower |
| GPU calls per iteration | `2841-3286` | `3671-3986` | Technique D more calls |
| Peak RAM iter 5 | `2620.96 MB` | `2402.82 MB` | Technique D lower RAM |
| Win rate vs greedy | `1.0` | `0.3` | Technique D worse |

## Historical Analysis

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

## Historical Improvements

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

## Historical Drawbacks

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

## Recommendation After Historical Run

The historical Technique D run should be reported as a negative multiprocessing result.

The better architecture to test next was:

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

That design avoids multiple CUDA contexts and preserves larger global batches. It is more complex, but it was the correct next direction if multiprocessing was continued.

For this project, Technique C remains the best implementation because it achieves high batch size, high MCTS throughput, and baseline-quality win rate without multiprocessing overhead.

## Implementation Update

The Technique D implementation has been revised to follow the better architecture above:

- self-play workers now run CPU-side MCTS traversal only
- the main process owns the PyTorch model during self-play
- workers send pending leaf boards to a parent inference queue
- the parent coalesces requests from all workers into one batched model call
- the parent keeps a lightweight inference cache, including symmetric positions, to reduce duplicate GPU work across workers
- `gpu_calls_per_iter` and `avg_gpu_batch_size` now report actual parent inference calls
- `numEps` is distributed across `numWorkers` with remainder handling instead of always using `args.numEps // 2`

## Revised Architecture Metrics: Parent-Owned Inference

The revised implementation was run with CPU-side self-play workers and parent-owned batched GPU inference:

```json
{
  "self_play_sec": [74.75, 76.97, 79.62, 76.25, 76.92],
  "train_sec": [41.41, 71.09, 111.01, 147.99, 185.56],
  "arena_sec": [94.70, 94.21, 92.39, 77.09, 73.81],
  "gpu_utilization_pct": [37.14, 43.40, 48.53, 54.07, 58.41],
  "mcts_sims_per_sec": [726.20, 705.77, 686.17, 718.79, 710.26],
  "peak_ram_mb": [1941.75, 2226.48, 2227.07, 2525.87, 2525.87],
  "cache_hit_rate_per_iter": [9.28, 12.07, 10.95, 12.81, 13.71],
  "gpu_calls_per_iter": [3503, 3256, 3327, 3014, 3127],
  "avg_gpu_batch_size": [11.53, 9.92, 9.11, 9.23, 9.14],
  "avg_virtual_loss_collisions_avoided": [0.048, 0.066, 0.078, 0.068, 0.070],
  "worker_utilization": [90.68, 93.88, 93.55, 91.65, 94.02],
  "examples_per_worker": [
    [6216, 6192],
    [6184, 6232],
    [6232, 6256],
    [6264, 6264],
    [6224, 6264]
  ],
  "win_rate_vs_greedy": 0.5
}
```

## Revised Summary Statistics

| Metric | Average | Range |
|--------|---------|-------|
| Self-play time | `76.90 sec` | `74.75-79.62 sec` |
| Training time | `111.41 sec` | `41.41-185.56 sec` |
| Arena time | `86.44 sec` | `73.81-94.70 sec` |
| GPU utilization | `48.31%` | `37.14-58.41%` |
| MCTS sims/sec | `709.44` | `686.17-726.20` |
| Peak RAM | `2289.41 MB` | `1941.75-2525.87 MB` |
| Cache hit rate | `11.76%` | `9.28-13.71%` |
| GPU calls per iter | `3245.40` | `3014-3503` |
| Avg GPU batch size | `9.79` | `9.11-11.53` |
| Worker utilization | `92.76%` | `90.68-94.02%` |
| Avg virtual-loss diversions | `0.066` | `0.048-0.078` |

Average total iteration time was `274.76 sec`, increasing from `210.86 sec` in iteration 1 to `336.29 sec` in iteration 5.

## Revised Comparison

| Metric | Technique C Final | Historical D | Revised D | Result |
|--------|-------------------|--------------|-----------|--------|
| Avg self-play time | ~`59.83 sec` | ~`60.68 sec` | `76.90 sec` | Revised D slower |
| Avg MCTS sims/sec | ~`1683.67` | ~`902.87` | `709.44` | Revised D lowest |
| Avg GPU utilization | ~`50.61%` | ~`52.27%` | `48.31%` | Revised D slightly lower |
| Avg GPU batch size | ~`11.17` | ~`9.35` | `9.79` | Revised D better than historical D, below C |
| Avg GPU calls per iter | ~`2980.40` | ~`3783.80` | `3245.40` | Revised D improves historical D, still above C |
| Peak RAM iter 5 | `2620.96 MB` | `2402.82 MB` | `2525.87 MB` | Revised D below C |
| Worker utilization | N/A | ~`90.64%` | `92.76%` | Workers active |
| Win rate vs greedy | `1.0` | `0.3` | `0.5` | Revised D improved but still worse than C |

## Revised Analysis

The revised architecture fixed the main design flaw from the historical run: workers no longer compete with separate CUDA contexts, and the main process now owns the GPU inference path. This improved some GPU-side indicators:

1. **GPU call count improved versus historical D**
   - Historical D averaged about `3784` GPU calls per iteration.
   - Revised D averaged about `3245`, so parent-side coalescing and caching helped.

2. **Average batch size improved versus historical D**
   - Historical D averaged about `9.35`.
   - Revised D averaged about `9.79`.
   - This confirms that parent-owned inference forms somewhat better global batches than two independent worker inference streams.

3. **Worker utilization stayed high**
   - Worker utilization averaged `92.76%`.
   - Example counts remained balanced between the two workers.
   - The slowdown is not caused by idle workers or bad episode splitting.

However, the revised design still does not improve end-to-end self-play speed:

1. **IPC latency moved into the MCTS hot path**
   - Workers now block on parent inference responses for leaf expansion.
   - Every neural leaf evaluation crosses process boundaries.
   - On Colab's 2-core CPU, queue serialization, scheduling, and synchronization can cost more than the saved CUDA contention.

2. **CPU-side MCTS throughput got worse**
   - Revised D averaged only `709` MCTS sims/sec.
   - Historical D averaged about `903`.
   - Technique C averaged about `1684`.
   - This confirms the bottleneck is Python-side traversal, game logic, and IPC coordination, not raw GPU inference.

3. **Global batching is still weaker than Technique C**
   - Revised D average batch size was `9.79`, below Technique C's `11.17`.
   - GPU calls were still higher than Technique C.
   - Parent coalescing helps, but the request/response rhythm does not match the simpler in-process lockstep loop.

4. **Quality improved but did not recover**
   - Win rate improved from historical D's `0.3` to `0.5`.
   - It still missed Technique C's `1.0`, so the revised architecture is not yet quality-equivalent.

## Updated Recommendation

Technique D remains a negative multiprocessing result for this Colab T4 setup.

The revised architecture is conceptually better than two CUDA workers and should be preferred if multiprocessing is kept, but the measured run shows that process IPC and CPU-side MCTS overhead dominate. For the current project constraints, Technique C remains the best result because it is simpler, faster, and quality-preserving.

Further Technique D work should only continue after profiling the revised hot path:

- parent inference queue wait time
- request serialization/deserialization time
- worker time spent waiting for inference responses
- `_select_action` and virtual-loss bookkeeping time
- `getValidMoves`, `getNextState`, `getGameEnded`, and `stringRepresentation` time

If the goal is speed on 2-core Colab, focus on optimizing Technique C's in-process CPU hot path before adding more multiprocessing. If the goal is distributed self-play on a larger CPU machine, Technique D may become useful with more workers, larger batches, and lower relative IPC overhead.

## Conclusion

Technique D successfully demonstrates two-process parallel self-play with Colab-safe `spawn`, balanced worker output, iteration checkpointing, and parent-owned GPU inference. The revised architecture fixed CUDA contention, reduced GPU calls compared with the historical D run, and improved win rate from `0.3` to `0.5`.

It still fails the main performance goal. Revised D self-play averaged `76.90 sec`, MCTS throughput dropped to `709` sims/sec, and quality remained below Technique C.

Final verdict: functional and architecturally cleaner than historical D, but still not recommended for Colab T4. Technique C should remain the primary optimization.
