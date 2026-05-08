# Technique C Scalability Testing Guide

## Overview

The `test_technique_c_scalability.py` script comprehensively tests Technique C (Lockstep Batched Self-Play) performance across various configurations to understand scalability characteristics.

## Features Tested

1. **Batch Size Scaling** - Tests how performance varies with different numbers of concurrent games (4, 8, 16, 24, 32)
   - Measures GPU batch efficiency
   - Tracks throughput improvement
   - Monitors cache hit rates

2. **Number of Games Scaling** - Tests performance with different total game counts (50, 100, 200, 400)
   - Measures stability across workload sizes
   - Tracks time and throughput scaling

3. **MCTS Simulations Scaling** - Tests performance with varying simulation depths (10, 25, 35, 50, 100)
   - Measures computation throughput (sims/sec)
   - Tracks game completion time
   - Shows GPU utilization patterns

4. **Board Size Scaling** - Tests different Othello board sizes (4x4, 6x6, 8x8)
   - Measures impact of state space complexity
   - Tracks memory consumption
   - Shows computational scaling

## Metrics Collected

For each test configuration, the script records:

- **Performance Metrics**
  - Games per second (throughput)
  - Time per game
  - MCTS simulations per second
  - Total elapsed time

- **Resource Metrics**
  - Peak GPU memory usage
  - Average GPU batch size
  - Cache hit/miss rates
  - Virtual loss diversions

## Usage

### Run All Tests (Default)

```bash
python test_technique_c_scalability.py
```

### Run Specific Test

```bash
# Test batch size scaling only
python test_technique_c_scalability.py --test-batch-size

# Test number of games scaling
python test_technique_c_scalability.py --test-num-games

# Test MCTS simulations scaling
python test_technique_c_scalability.py --test-mcts-sims

# Test board size scaling
python test_technique_c_scalability.py --test-board-size
```

### Specify Output Directory

```bash
python test_technique_c_scalability.py --output-dir ./my_results
```

### Run All Tests with Custom Output

```bash
python test_technique_c_scalability.py --all --output-dir ./technique_c_benchmarks
```

## Output

Results are saved as JSON files in the output directory with timestamp:

- `technique_c_scalability_YYYYMMDD_HHMMSS.json`

Each result includes:

```json
{
  "test_name": "batch_size_16",
  "timestamp": "2026-05-08T14:30:45.123456",
  "batch_size": 16,
  "num_games": 100,
  "num_mcts_sims": 35,
  "board_size": 6,
  "elapsed_time": 45.23,
  "games_per_sec": 2.21,
  "time_per_game": 0.452,
  "total_mcts_sims": 3500,
  "mcts_sims_per_sec": 77.4,
  "peak_memory_mb": 2048.5,
  "avg_gpu_batch_size": 15.2,
  "cache_hit_rate": 0.65,
  ...
}
```

## Customization

To modify test parameters, edit the default arguments in `main()`:

```python
# Modify batch sizes to test
tester.test_batch_size_scaling(
    batch_sizes=(2, 4, 8, 16, 32, 64),  # Custom sizes
    num_games=150,
    num_sims=50,
    board_size=6
)
```

## Interpretation

### Batch Size Results

- **Look for:** Throughput improvement as batch size increases, then plateau
- **Optimal:** Batch size where GPU utilization peaks without diminishing returns
- **Trade-off:** Larger batches use more memory

### Number of Games Results

- **Look for:** Consistent throughput across different workload sizes
- **Plateau indicates:** System is saturated at that configuration

### MCTS Simulations Results

- **Look for:** Linear or sub-linear increase in time with simulation count
- **Higher sims/sec:** More efficient GPU batching for deeper searches

### Board Size Results

- **Look for:** Impact of computational complexity on throughput
- **Memory scaling:** How memory usage grows with board complexity

## Requirements

- PyTorch with CUDA support (for GPU testing)
- Othello game implementation
- Batched self-play module
- Neural network wrapper

## Tips

1. **Baseline Run** - Run once with `--test-mcts-sims` to establish baseline performance
2. **Warm-up** - Allow GPU to warm up before running critical tests
3. **Comparison** - Compare results before/after optimization changes
4. **Logging** - Check logs for warnings about missing metrics or simulation issues
