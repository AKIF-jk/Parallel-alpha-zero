#!/usr/bin/env python3
"""
Example usage scenarios for Technique C scalability testing.

This file demonstrates various ways to use the scalability tester
for different analysis goals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_technique_c_scalability import TechniqueCScalabilityTester


def example_1_find_optimal_batch_size():
    """Find the optimal batch size for GPU utilization."""
    print("Example 1: Finding Optimal Batch Size")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/batch_size_tuning')
    
    # Test a wide range of batch sizes with fixed game/sim count
    tester.test_batch_size_scaling(
        batch_sizes=(2, 4, 8, 12, 16, 20, 24, 32, 40, 48),
        num_games=200,
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
    
    # Test with increasing numbers of games
    tester.test_num_games_scaling(
        num_games_list=(25, 50, 100, 200, 400, 800),
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
    
    # Test shallow to deep simulations
    tester.test_mcts_sims_scaling(
        num_sims_list=(5, 10, 15, 25, 35, 50, 75, 100, 150),
        num_games=100,
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
    
    # Test different board sizes (state space complexity)
    tester.test_board_size_scaling(
        board_sizes=(4, 5, 6, 7, 8),
        num_games=100,
        batch_size=16,
        num_sims=35
    )
    
    tester.print_summary()
    tester.save_results()


def example_5_comprehensive_benchmark():
    """Run comprehensive benchmark across all dimensions."""
    print("\nExample 5: Comprehensive Benchmark")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/comprehensive')
    
    # Run all test types with moderate configurations
    tester.test_batch_size_scaling(
        batch_sizes=(8, 16, 24, 32),
        num_games=100,
        num_sims=35,
        board_size=6
    )
    
    tester.test_num_games_scaling(
        num_games_list=(50, 100, 200),
        batch_size=16,
        num_sims=35,
        board_size=6
    )
    
    tester.test_mcts_sims_scaling(
        num_sims_list=(20, 35, 50),
        num_games=100,
        batch_size=16,
        board_size=6
    )
    
    tester.test_board_size_scaling(
        board_sizes=(6, 8),
        num_games=100,
        batch_size=16,
        num_sims=35
    )
    
    tester.print_summary()
    tester.save_results()


def example_6_memory_constrained_testing():
    """Test configurations suitable for memory-constrained environments."""
    print("\nExample 6: Memory-Constrained Testing")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/memory_constrained')
    
    # Test with small batch sizes and smaller boards
    tester.test_batch_size_scaling(
        batch_sizes=(2, 4, 8, 12),
        num_games=50,
        num_sims=25,
        board_size=4
    )
    
    tester.test_board_size_scaling(
        board_sizes=(4, 5, 6),
        num_games=50,
        batch_size=4,
        num_sims=25
    )
    
    tester.print_summary()
    tester.save_results()


def example_7_high_throughput_tuning():
    """Tune for maximum throughput on high-end hardware."""
    print("\nExample 7: High-Throughput Tuning")
    print("-" * 60)
    
    tester = TechniqueCScalabilityTester(output_dir='results/high_throughput')
    
    # Test with large batch sizes and many games
    tester.test_batch_size_scaling(
        batch_sizes=(32, 48, 64, 96, 128),
        num_games=500,
        num_sims=50,
        board_size=6
    )
    
    tester.test_mcts_sims_scaling(
        num_sims_list=(35, 50, 75, 100),
        num_games=300,
        batch_size=64,
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
        choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
        default='all',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    examples = {
        '1': example_1_find_optimal_batch_size,
        '2': example_2_assess_throughput_stability,
        '3': example_3_benchmark_across_sim_depths,
        '4': example_4_board_complexity_impact,
        '5': example_5_comprehensive_benchmark,
        '6': example_6_memory_constrained_testing,
        '7': example_7_high_throughput_tuning,
    }
    
    if args.example == 'all':
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
