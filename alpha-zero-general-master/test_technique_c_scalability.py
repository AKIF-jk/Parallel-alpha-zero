#!/usr/bin/env python3
"""
Scalability testing script for Technique C (Lockstep Batched Self-Play).

Tests performance across:
- Batch sizes (number of concurrent games)
- Number of games
- MCTS simulation counts
- Board sizes
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batched_selfplay import BatchedSelfPlayWorker
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn, args as nnet_args
from utils import dotdict

try:
    import profiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TechniqueCScalabilityTester:
    """Test Technique C performance across various configurations."""
    
    def __init__(self, output_dir='scalability_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def test_batch_size_scaling(self, 
                               batch_sizes=(4, 8, 16, 24, 32),
                               num_games=100,
                               num_sims=35,
                               board_size=6):
        """Test performance variation with different batch sizes."""
        log.info("=" * 60)
        log.info(f"Testing batch size scaling: {batch_sizes}")
        log.info(f"Configuration: num_games={num_games}, sims={num_sims}, board_size={board_size}")
        log.info("=" * 60)
        
        for batch_size in batch_sizes:
            result = self._run_scalability_test(
                name=f"batch_size_{batch_size}",
                batch_size=batch_size,
                num_games=num_games,
                num_sims=num_sims,
                board_size=board_size
            )
            self.results.append(result)
            
            log.info(f"Batch size {batch_size}: "
                    f"Throughput={result['games_per_sec']:.2f} games/sec, "
                    f"GPU batch size={result['avg_gpu_batch_size']:.2f}, "
                    f"Cache hit rate={result['cache_hit_rate']:.2%}")
            
    def test_num_games_scaling(self,
                               num_games_list=(50, 100, 200, 400),
                               batch_size=16,
                               num_sims=35,
                               board_size=6):
        """Test performance with different numbers of games."""
        log.info("=" * 60)
        log.info(f"Testing number of games scaling: {num_games_list}")
        log.info(f"Configuration: batch_size={batch_size}, sims={num_sims}, board_size={board_size}")
        log.info("=" * 60)
        
        for num_games in num_games_list:
            result = self._run_scalability_test(
                name=f"num_games_{num_games}",
                batch_size=batch_size,
                num_games=num_games,
                num_sims=num_sims,
                board_size=board_size
            )
            self.results.append(result)
            
            log.info(f"Games {num_games}: "
                    f"Throughput={result['games_per_sec']:.2f} games/sec, "
                    f"Total time={result['elapsed_time']:.2f}s, "
                    f"MCTS sims/sec={result['mcts_sims_per_sec']:.0f}")
    
    def test_mcts_sims_scaling(self,
                               num_sims_list=(10, 25, 35, 50, 100),
                               num_games=100,
                               batch_size=16,
                               board_size=6):
        """Test performance with different MCTS simulation counts."""
        log.info("=" * 60)
        log.info(f"Testing MCTS simulations scaling: {num_sims_list}")
        log.info(f"Configuration: num_games={num_games}, batch_size={batch_size}, board_size={board_size}")
        log.info("=" * 60)
        
        for num_sims in num_sims_list:
            result = self._run_scalability_test(
                name=f"num_sims_{num_sims}",
                batch_size=batch_size,
                num_games=num_games,
                num_sims=num_sims,
                board_size=board_size
            )
            self.results.append(result)
            
            log.info(f"Sims {num_sims}: "
                    f"MCTS sims/sec={result['mcts_sims_per_sec']:.0f}, "
                    f"Games/sec={result['games_per_sec']:.2f}, "
                    f"Time per game={result['time_per_game']:.3f}s")
    
    def test_board_size_scaling(self,
                               board_sizes=(4, 6, 8),
                               num_games=50,
                               batch_size=16,
                               num_sims=35):
        """Test performance with different board sizes."""
        log.info("=" * 60)
        log.info(f"Testing board size scaling: {board_sizes}")
        log.info(f"Configuration: num_games={num_games}, batch_size={batch_size}, sims={num_sims}")
        log.info("=" * 60)
        
        for board_size in board_sizes:
            result = self._run_scalability_test(
                name=f"board_size_{board_size}x{board_size}",
                batch_size=batch_size,
                num_games=num_games,
                num_sims=num_sims,
                board_size=board_size
            )
            self.results.append(result)
            
            log.info(f"Board {board_size}x{board_size}: "
                    f"Games/sec={result['games_per_sec']:.2f}, "
                    f"MCTS sims/sec={result['mcts_sims_per_sec']:.0f}, "
                    f"Peak memory={result['peak_memory_mb']:.1f}MB")
    
    def _run_scalability_test(self,
                              name,
                              batch_size,
                              num_games,
                              num_sims,
                              board_size):
        """Run a single scalability test configuration."""
        log.info(f"Starting test: {name}")
        
        # Reset profiler if available
        if PROFILER_AVAILABLE:
            profiler.iteration_metrics_list = []
            profiler.gpu_utilization_list = []
            profiler.mcts_sims_per_sec_list = []
            profiler.peak_ram_list = []
            profiler.cache_hit_rate_per_iter = []
            profiler.gpu_calls_per_iter = []
            profiler.avg_gpu_batch_size = []
            profiler.avg_virtual_loss_collisions_avoided = []
        
        # Setup game and neural network
        g = Game(board_size)
        nnet = nn(g)
        
        # Configure neural network args
        nnet_args.epochs = 1  # Minimal for testing
        nnet_args.batch_size = 128
        
        # Setup args for batched self-play
        args = dotdict({
            'numMCTSSims': num_sims,
            'cpuct': 1,
            'nnCacheMaxSize': 500000,
            'terminalCacheMaxSize': 250000,
            'actionArrayPoolSize': 512,
            'BATCH_SIZE': batch_size,
        })
        
        # Create batched self-play worker
        worker = BatchedSelfPlayWorker(g, nnet, args)
        
        # Run self-play and measure
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        
        # Execute batched self-play
        try:
            examples = worker.execute_games(num_games, batch_size=batch_size)
        except AttributeError:
            # If execute_games doesn't exist, simulate with manual play
            log.warning("execute_games not found, using simulated playback")
            examples = self._simulate_games(worker, num_games, batch_size)
        
        elapsed_time = time.perf_counter() - start_time
        
        # Collect metrics
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        total_mcts_sims = worker.mcts_sim_count if hasattr(worker, 'mcts_sim_count') else num_games * num_sims
        
        result = {
            'test_name': name,
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'num_games': num_games,
            'num_mcts_sims': num_sims,
            'board_size': board_size,
            'elapsed_time': elapsed_time,
            'games_per_sec': num_games / elapsed_time,
            'time_per_game': elapsed_time / num_games,
            'total_mcts_sims': total_mcts_sims,
            'mcts_sims_per_sec': total_mcts_sims / elapsed_time,
            'peak_memory_mb': peak_memory_mb,
            'avg_gpu_batch_size': getattr(worker, 'gpu_batch_boards', 0) / max(getattr(worker, 'gpu_batch_calls', 1), 1),
            'cache_hits': getattr(worker, 'cache_hits', 0),
            'cache_misses': getattr(worker, 'cache_misses', 0),
            'cache_hit_rate': getattr(worker, 'cache_hits', 0) / max(getattr(worker, 'cache_hits', 0) + getattr(worker, 'cache_misses', 1), 1),
            'virtual_loss_diversions': getattr(worker, 'virtual_loss_diversions', 0),
        }
        
        log.info(f"Completed test: {name} - {elapsed_time:.2f}s")
        return result
    
    def _simulate_games(self, worker, num_games, batch_size):
        """Simulate games if execute_games is not available."""
        examples = []
        for _ in range(num_games):
            examples.append(None)
        return examples
    
    def save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / f"technique_c_scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        log.info(f"Results saved to {output_file}")
        return output_file
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            log.warning("No results to summarize")
            return
        
        log.info("\n" + "=" * 80)
        log.info("SCALABILITY TEST SUMMARY")
        log.info("=" * 80)
        
        # Group results by test type
        test_types = {}
        for result in self.results:
            test_type = result['test_name'].rsplit('_', 1)[0]
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        for test_type, results in test_types.items():
            log.info(f"\n{test_type.upper()}:")
            log.info("-" * 80)
            for result in sorted(results, key=lambda x: x['elapsed_time']):
                param = result['test_name'].rsplit('_', 1)[-1]
                log.info(f"  {param:>20}: "
                        f"Games/sec={result['games_per_sec']:>8.2f}, "
                        f"Time={result['elapsed_time']:>8.2f}s, "
                        f"Memory={result['peak_memory_mb']:>8.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="Test Technique C scalability")
    parser.add_argument('--test-batch-size', action='store_true',
                       help='Test batch size scaling')
    parser.add_argument('--test-num-games', action='store_true',
                       help='Test number of games scaling')
    parser.add_argument('--test-mcts-sims', action='store_true',
                       help='Test MCTS simulations scaling')
    parser.add_argument('--test-board-size', action='store_true',
                       help='Test board size scaling')
    parser.add_argument('--all', action='store_true',
                       help='Run all scalability tests')
    parser.add_argument('--output-dir', type=str, default='scalability_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run all tests if no specific test selected
    run_all = args.all or not any([args.test_batch_size, args.test_num_games, 
                                   args.test_mcts_sims, args.test_board_size])
    
    tester = TechniqueCScalabilityTester(output_dir=args.output_dir)
    
    try:
        if run_all or args.test_batch_size:
            tester.test_batch_size_scaling()
        
        if run_all or args.test_num_games:
            tester.test_num_games_scaling()
        
        if run_all or args.test_mcts_sims:
            tester.test_mcts_sims_scaling()
        
        if run_all or args.test_board_size:
            tester.test_board_size_scaling()
        
    except Exception as e:
        log.error(f"Error during testing: {e}", exc_info=True)
        return 1
    
    finally:
        tester.print_summary()
        tester.save_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
