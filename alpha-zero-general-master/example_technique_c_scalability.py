#!/usr/bin/env python3
"""
Example usage scenarios for real Technique C scalability testing.

The examples below run actual BatchedSelfPlayWorker.execute_batch() calls.
They are sized for Colab T4-style experiments, so run one focused example at
a time unless you intentionally want a longer benchmark sweep.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_technique_c_scalability import TechniqueCScalabilityTester


def example_0_smoke_test():
    """Run a short sanity check before longer sweeps."""
    print("Example 0: Smoke Test")
    print("-" * 60)

    tester = TechniqueCScalabilityTester(output_dir='results/smoke_test')

    tester.test_batch_size_scaling(
        batch_sizes=(4, 8, 16),
        num_games=16,
        num_sims=10,
        board_size=6
    )

    tester.print_summary()
    tester.save_results()


def example_1_find_optimal_batch_size():
    """Find the optimal batch size for GPU utilization."""
    print("Example 1: Finding Optimal Batch Size")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/batch_size_tuning')
    
    # Focused T4 sweep. Larger values are useful only after this finishes cleanly.
    tester.test_batch_size_scaling(
        batch_sizes=(4, 8, 12, 16, 24, 32),
        num_games=96,
        num_sims=35,
        board_size=6
    )
    
    tester.print_summary()
    tester.save_results()


def example_2_assess_throughput_stability():
    """Assess how stable throughput is across different workload sizes."""
    print("\nExample 2: Assessing Throughput Stability")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/throughput_stability')
    
    # Keep batch and sim count fixed; vary total work to check stability.
    tester.test_num_games_scaling(
        num_games_list=(16, 32, 48, 96, 192),
        batch_size=16,
        num_sims=35,
        board_size=6
    )
    
    tester.print_summary()
    tester.save_results()


def example_3_benchmark_across_sim_depths():
    """Benchmark performance with different MCTS simulation depths."""
    print("\nExample 3: MCTS Simulation Depth Benchmarking")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/mcts_depth_benchmark')
    
    # Deeper searches cost more, but should usually improve batching stability.
    tester.test_mcts_sims_scaling(
        num_sims_list=(10, 20, 35, 50, 75),
        num_games=48,
        batch_size=16,
        board_size=6
    )
    
    tester.print_summary()
    tester.save_results()


def example_4_board_complexity_impact():
    """Measure impact of board complexity on performance."""
    print("\nExample 4: Board Complexity Impact")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/board_complexity')
    
    # Use even board sizes. The C++ bitboard backend is not enabled here.
    tester.test_board_size_scaling(
        board_sizes=(4, 6, 8),
        num_games=32,
        batch_size=16,
        num_sims=20
    )
    
    tester.print_summary()
    tester.save_results()


def example_5_comprehensive_benchmark():
    """Run comprehensive benchmark across all dimensions."""
    print("\nExample 5: Comprehensive Benchmark")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/comprehensive')
    
    # A compact all-around run that is suitable for reporting trends.
    tester.test_batch_size_scaling(
        batch_sizes=(8, 16, 24),
        num_games=48,
        num_sims=35,
        board_size=6
    )
    
    tester.test_num_games_scaling(
        num_games_list=(32, 48, 96),
        batch_size=16,
        num_sims=35,
        board_size=6
    )
    
    tester.test_mcts_sims_scaling(
        num_sims_list=(20, 35, 50),
        num_games=48,
        batch_size=16,
        board_size=6
    )
    
    tester.test_board_size_scaling(
        board_sizes=(6, 8),
        num_games=32,
        batch_size=16,
        num_sims=20
    )
    
    tester.print_summary()
    tester.save_results()


def example_6_memory_constrained_testing():
    """Test configurations suitable for memory-constrained environments."""
    print("\nExample 6: Memory-Constrained Testing")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/memory_constrained')
    
    # Small workloads for low-memory sessions.
    tester.test_batch_size_scaling(
        batch_sizes=(2, 4, 8),
        num_games=24,
        num_sims=15,
        board_size=4
    )
    
    tester.test_board_size_scaling(
        board_sizes=(4, 6),
        num_games=24,
        batch_size=4,
        num_sims=15
    )
    
    tester.print_summary()
    tester.save_results()


def example_7_high_throughput_tuning():
    """Tune larger batches after the smaller sweeps are validated."""
    print("\nExample 7: Larger-Batch Tuning")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/larger_batch_tuning')
    
    # This is intentionally heavier than the other examples.
    tester.test_batch_size_scaling(
        batch_sizes=(16, 24, 32, 48, 64),
        num_games=128,
        num_sims=50,
        board_size=6
    )
    
    tester.test_mcts_sims_scaling(
        num_sims_list=(35, 50, 75),
        num_games=96,
        batch_size=32,
        board_size=6
    )
    
    tester.print_summary()
    tester.save_results()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Example usage of Technique C scalability tester"
    )
    parser.add_argument(
        'example',
        nargs='?',
        choices=['0', '1', '2', '3', '4', '5', '6', '7', 'all'],
        default='0',
        help='Which example to run. Default is the short smoke test.'
    )
    
    args = parser.parse_args()
    
    examples = {
        '0': example_0_smoke_test,
        '1': example_1_find_optimal_batch_size,
        '2': example_2_assess_throughput_stability,
        '3': example_3_benchmark_across_sim_depths,
        '4': example_4_board_complexity_impact,
        '5': example_5_comprehensive_benchmark,
        '6': example_6_memory_constrained_testing,
        '7': example_7_high_throughput_tuning,
    }
    
    if args.example == 'all':
        print("Running all examples can take a long time on Colab T4.")
        for example_fn in examples.values():
            try:
                example_fn()
            except Exception as e:
                print(f"Error in example: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            examples[args.example]()
        except Exception as e:
            print(f"Error in example: {e}")
            import traceback
            traceback.print_exc()
