# Unified Milestone 3 Report

## 1. Introduction

This project studies performance optimization of AlphaZero-style self-play training for Othello under constrained compute (Google Colab T4). The base objective was not only to preserve playing quality but to improve throughput, efficiency, and practical runtime using techniques grounded in Parallel and Distributed Computing (PDC): batching, cache-aware design, memory-aware data structures, and multiprocessing.

The baseline implementation was progressively extended through five experimental techniques (A to E), plus phased integration work from an optimization roadmap. Each change was evaluated empirically with profiling metrics including self-play time, MCTS simulations per second, GPU utilization, memory usage, GPU call behavior, and win rate against greedy evaluation.

## 2. Base Paper and Problem Context

The selected paper, Learning to Play Othello Without Human Knowledge (Thakoor, Nair, Jhunjhunwala), adapts AlphaGo Zero ideas to Othello. The key pattern is iterative policy improvement via:

- self-play episodes
- MCTS-guided action selection
- neural network training on generated examples
- periodic model selection through arena evaluation

In this framework, the largest practical costs come from repeated MCTS traversal, game-state transitions, and frequent neural network inference. For resource-constrained environments, the central systems question is how to reduce overhead per useful simulation while preserving the policy/value learning signal.

## 3. Project Scope and Objectives

### Scope

- Platform: alpha-zero-general Othello codebase
- Target: end-to-end training loop (self-play, train, arena)
- Constraint: Colab-class hardware and limited runtime window

### Objectives

1. Improve MCTS throughput without harming learning quality.
2. Increase GPU efficiency by reducing small/fragmented inference calls.
3. Evaluate memory-performance tradeoffs of caches and data structures.
4. Test whether multiprocessing improves performance in this workload.
5. Produce reproducible, metrics-driven conclusions for each technique.

### Success Criteria

- Higher MCTS simulations per second
- Lower self-play wall-clock per equivalent setting
- Better GPU batching behavior (larger average batch, fewer calls)
- Quality preserved (win_rate_vs_greedy close to or equal to strong baseline)

## 4. Baseline Method

Baseline corresponds to standard single-process AlphaZero-style training in this repository:

- per-episode self-play with MCTS
- dictionary-based MCTS statistics
- direct neural inference during leaf expansion
- no explicit cross-episode batched inference coordinator

Reported baseline behavior (from baseline metrics and comparisons used in technique reports):

- MCTS throughput around 1075 to 1151 sims/sec
- strong greedy quality benchmark (win rate near 1.0 in baseline comparisons)
- moderate RAM growth across iterations

This baseline serves as the reference for speedup and quality retention.

## 5. Proposed Parallel / Optimized Approach

This section summarizes each implemented technique, why it was expected to help, what changed, and what actually happened.

### Technique A: Numpy Arrays for MCTS State Storage

#### Hypothesis (why it should help)

A contiguous array-based node layout should improve cache locality and speed up UCB computations through vectorized operations.

#### Implementation

- Replaced multiple MCTS dictionaries with node objects containing numpy arrays (Q, N, P, valid mask, terminal flags).
- Vectorized UCB computation and argmax action selection.

#### Outcome

- Quality preserved (win rate remained 1.0 in reported comparison).
- Performance regressed by about 5 to 10 percent versus baseline for 6x6.

#### Why positive change did not appear

1. Action space is small (37 actions on 6x6), so numpy call overhead dominates vectorization gains.
2. Extra node object and array allocations add overhead.
3. Access path changed from one dict lookup to dict plus attribute plus array indexing.
4. For small vectors, Python-loop/dict paths can outperform numpy-heavy dispatch.

#### Verdict

Correct but not performance-beneficial in this workload.

### Technique B: Neural Network Position Cache for MCTS

#### Hypothesis

Many board states recur (including symmetries). Caching policy/value predictions should reduce repeated GPU calls and speed self-play.

#### Implementation

- Added neural inference cache in MCTS keyed by board representation.
- Added symmetry-aware cache insertion.
- Reset search tree per episode but kept cache across episodes inside an iteration.
- Reset cache at iteration start to avoid stale predictions after weight updates.

#### Improvements observed

- Self-play average improved (about 5.5 percent faster than Technique A).
- MCTS throughput improved to around Technique B average 1076 sims/sec.
- GPU calls reduced over iterations as hit rate rose.
- Cache hit rate increased from about 12.89 percent to 19.44 percent.

#### Drawbacks and uncertainty

- Higher peak RAM versus prior run due to cache growth.
- Win rate vs greedy in that run was 0.6 (below Technique A/C outcomes).

#### Interpretation

Technique B is a real systems improvement for throughput and GPU-call reduction, but quality confidence requires longer/fixed-seed evaluation.

### Technique C: Lockstep Batched Self-Play (Single Process)

#### Hypothesis

Coordinating many active games in lockstep allows leaf requests from different games to be merged into larger GPU batches, reducing inference fragmentation and improving overall throughput.

#### Implementation

- Introduced BatchedSelfPlayWorker with rolling-refill active slots (batch size 16).
- Shared search-state dictionaries across active games for transposition reuse.
- Pending leaf coalescing for duplicate board requests in the same batch.
- Kept NN cache behavior and added terminal cache.
- Added virtual visit/loss style coordination and stuck-game guard.

#### Improvements observed (final strong run)

- MCTS throughput increased strongly (about 1605 to 1848 sims/sec, average around 1684).
- Much fewer GPU calls per iteration (about 2841 to 3286) compared with prior fragmented inference behavior.
- Healthy average GPU batch size (about 9.43 to 12.68, near practical target for 16 active slots).
- Win rate vs greedy reached 1.0 under stronger settings.

#### Tradeoffs

- Higher RAM usage (peak near 2621 MB).
- Longer full-iteration time under stronger training/evaluation settings (mainly due to larger experiment settings, not batching failure).

#### Verdict

Best overall technique in this project: strong throughput gain, good batching behavior, quality preserved.

### Technique D: Two-Process Parallel Batched Self-Play

#### Initial hypothesis

Using two worker processes should exploit available CPU cores and accelerate self-play.

#### Historical design

- Two workers each performed batched self-play and CUDA inference.

#### Why this seemed beneficial

Parallel workers should increase concurrent MCTS work and potentially improve hardware utilization.

#### What happened (historical result)

- No self-play speedup vs Technique C.
- Throughput dropped substantially versus Technique C.
- Win rate degraded to 0.3.

#### Root causes for failure

1. Single-GPU contention from multiple CUDA contexts.
2. Multiprocessing spawn and IPC overhead.
3. Loss of global shared tree/cache reuse across all games.
4. Smaller effective batches per worker than one global coordinator.

#### Revised design

- Parent-owned GPU inference queue.
- Workers perform CPU-side traversal only and send leaf requests.
- Parent coalesces and caches inference requests.

#### Revised result

- Better than historical D on some GPU-side indicators (batch size and call count improved vs historical D).
- Still slower than Technique C overall, with lower MCTS sims/sec and lower quality (win rate 0.5).

#### Why positive change still did not appear

IPC latency and synchronization moved directly into the hot MCTS path. On low-core Colab settings, queue serialization/wait costs dominated gains.

#### Verdict

Technically functional and architecturally improved from historical D, but not a performance win for this hardware profile.

### Technique E: Batched Self-Play With Explicit Virtual Loss

#### Hypothesis

Explicit virtual-loss handling should reduce duplicate branch selection among concurrently active games and improve effective search efficiency.

#### Implementation

- Kept Technique C single-process batched architecture.
- Added explicit virtual-loss logic and diversion tracking in action selection.

#### Improvements observed

- Quality preserved: win_rate_vs_greedy remained 1.0.
- GPU batching remained healthy (avg batch about 11.25).
- Cache hit rate improved (average about 15.05 percent vs lower Technique C run average).
- GPU calls stayed low and competitive.

#### Why net speedup did not appear

1. Additional bookkeeping in the hottest selection path increased CPU cost.
2. Collision-avoidance activity was too small (average diversions about 0.067 per simulation) to repay overhead.
3. Bottleneck remained CPU-side traversal/game logic, not GPU batching.

#### Verdict

Quality-safe but speed-negative under current implementation; keep only minimal virtual-loss logic in hot path and gate expensive diagnostics behind profiling flags.

### Phase Integration Notes (Optimization Roadmap)

Integrated engineering work from the optimization plan included:

- bounded LRU caches for safer memory behavior
- array pooling for reduced temporary allocation overhead
- optional Zobrist hashing
- optional parallel MCTS thread-pool mode
- optional C++ bitboard backend

Observed benchmark summary from integrated phase testing:

- python_tobytes path remained fastest
- current Zobrist path slower for this workload
- bitboard C++ path improved over Zobrist but stayed below baseline because Python/C++ conversion overhead remained significant in hot loops

This supports a key systems insight: reducing interface overhead in hot loops matters as much as raw kernel speed.

## 6. Experimental Setup

### Hardware and Runtime Environment

- Google Colab environment
- NVIDIA T4 GPU
- CPU-limited runtime profile (noted 2-core setting in multiprocessing analysis)

### Software

- Python-based alpha-zero-general codebase
- PyTorch for neural policy/value network
- Othello 6x6 training focus for most optimization comparisons

### Common Strong-Run Parameters (used in final Technique C/E style evaluations)

- numIters: 5
- numEps: 48
- numMCTSSims: 35
- arenaCompare: 40
- greedyCompare: 40
- training epochs: 15
- training batch size: 128
- batched self-play active slots: 16

### Core Metrics Collected

- self_play_sec
- train_sec
- arena_sec
- gpu_utilization_pct
- mcts_sims_per_sec
- peak_ram_mb
- cache_hit_rate_per_iter
- gpu_calls_per_iter
- avg_gpu_batch_size
- win_rate_vs_greedy
- multiprocessing diagnostics when relevant (worker utilization, examples per worker)

## 7. Results and Discussion

### 7.1 Consolidated Comparison

| Technique | Main Idea                                      | Throughput Effect                               | Quality Effect              | Overall                                |
| --------- | ---------------------------------------------- | ----------------------------------------------- | --------------------------- | -------------------------------------- |
| Baseline  | Standard per-episode self-play + MCTS          | Reference (about 1075 to 1151 sims/sec)         | Strong                      | Reference                              |
| A         | Numpy node arrays for MCTS state               | Negative (about 5 to 10 percent slower)         | Preserved (1.0)             | Not recommended for 6x6                |
| B         | NN position cache + symmetry reuse             | Positive (self-play and sims/sec improved vs A) | Mixed (0.6 in reported run) | Keep with stronger quality validation  |
| C         | Lockstep batched self-play, single process     | Strong positive (about 1605 to 1848 sims/sec)   | Preserved (1.0)             | Best result                            |
| D         | Two-process self-play (historical and revised) | Negative vs C (both variants)                   | Degraded (0.3 then 0.5)     | Not recommended on Colab T4            |
| E         | Technique C + explicit virtual loss            | Slight negative vs C on throughput              | Preserved (1.0)             | Quality-safe, speed-negative currently |

### 7.2 Speedup and Efficiency Interpretation

- Largest practical speedup came from Technique C due to better batching and lower inference fragmentation.
- Technique B offered meaningful gains through inference reuse and reduced repeated GPU calls.
- Techniques D and E show that added coordination logic can increase overhead in the MCTS hot path when CPU resources are limited.
- Efficiency is workload-sensitive: optimizations that look favorable in theory (array vectorization, extra parallel orchestration) may fail when overheads dominate small action-space loops.

### 7.3 Scalability Discussion

- In-process batched coordination (Technique C) scaled well across active games under single-process constraints.
- Cross-process scaling (Technique D) did not translate to speedup on this machine profile because communication and synchronization costs dominated.
- Strong-run settings (more games/sims/epochs) improved final model quality but also increased runtime and memory, highlighting a clear quality-vs-cost tradeoff.

### 7.4 Bottleneck Analysis

Evidence across C, D, E suggests the main bottleneck after batching improvements is CPU-side MCTS and game logic:

- action selection bookkeeping
- game transition and validity checks
- state representation and hashing/lookup overhead
- Python-level coordination overhead

GPU path is reasonably optimized in C/E (healthy batch size, reduced call count), so further gains are likely to come from CPU hot-path optimization and low-overhead data flow design.

### 7.5 LLM Usage Disclosure and Reflection

LLM assistance was used for implementation planning, code refinement, profiling guidance, and report drafting. All generated suggestions were validated through empirical runs and metric checks.

Benefits:

- faster iteration on optimization ideas
- clearer experimental documentation
- easier comparison across multiple techniques

Limitations and risks:

- suggested optimizations can be plausible but workload-mismatched
- over-reliance can hide assumptions unless every change is benchmarked
- correctness and performance both require direct verification

Mitigation used in this project:

- metric-driven validation for each technique
- explicit reporting of negative results, not only positive outcomes
- preserving reproducibility through logged parameters and saved metrics

## 8. Conclusion

This project demonstrates that the most effective optimization for this AlphaZero-style Othello workload on Colab T4 is single-process lockstep batched self-play (Technique C). It delivered the best combination of throughput gain, reduced GPU call fragmentation, and preserved quality.

Not all theoretically promising ideas improved performance:

- Technique A (numpy-node layout) regressed due to overhead dominating small action-space loops.
- Technique D (multiprocessing) remained slower than C because IPC/synchronization costs dominated on limited CPU resources.
- Technique E (explicit virtual-loss bookkeeping) preserved quality but reduced throughput due to hot-path overhead.

Positive contributions retained:

- caching and symmetry reuse (Technique B)
- batched inference coordination and duplicate-leaf coalescing (Technique C)
- strong profiling instrumentation and reproducible comparisons

Final recommendation: use Technique C as the primary production path for this project, keep selected cache improvements, and prioritize CPU hot-path optimization before further multiprocessing complexity.

## 9. References

1. Thakoor, S., Nair, S., Jhunjhunwala, M. Learning to Play Othello Without Human Knowledge.
2. Silver, D. et al. Mastering the game of Go without human knowledge. Nature, 2017.
3. Browne, C. B. et al. A Survey of Monte Carlo Tree Search Methods. IEEE TCIAIG, 2012.
4. Project repository experiment logs and result documents:
   - technique_a_results.md
   - technique_b_results.md
   - technique_c_results.md
   - technique_d_results.md
   - technique_e_results.md
   - optimization_changes.md
   - project.txt
