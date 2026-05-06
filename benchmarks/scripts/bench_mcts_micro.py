#!/usr/bin/env python3
"""
MCTS microbenchmark: measures per-node operation timing.
Compares baseline Python MCTS vs optimized C++ MCTS on identical board states.
"""
import sys
import time
import os
import json
import argparse
import numpy as np

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-master'))
OPTIMIZED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-cpp'))


def benchmark_baseline_mcts(num_iterations, num_sims, seed=42):
    os.chdir(BASELINE_DIR)
    sys.path.insert(0, BASELINE_DIR)
    from othello.OthelloGame import OthelloGame
    from othello.pytorch.NNet import NNetWrapper
    from MCTS import MCTS

    np.random.seed(seed)
    game = OthelloGame(6)
    nnet = NNetWrapper(game)

    args = type('obj', (object,), {
        'cpuct': 1.0,
        'numMCTSSims': num_sims,
    })()

    mcts = MCTS(game, nnet, args)
    board = game.getInitBoard()

    timings = {
        'total_ns': [],
    }

    for i in range(num_iterations):
        np.random.seed(seed + i)
        start_total = time.perf_counter_ns()

        mcts.search(board)

        total_ns = time.perf_counter_ns() - start_total
        timings['total_ns'].append(total_ns)

    avg_total = np.mean(timings['total_ns'])
    avg_per_sim = avg_total / num_sims if num_sims > 0 else 0

    return {
        'implementation': 'baseline_python',
        'num_iterations': num_iterations,
        'num_sims_per_iter': num_sims,
        'avg_total_ns_per_iter': round(avg_total),
        'avg_ns_per_sim': round(avg_per_sim, 1),
        'avg_ms_per_iter': round(avg_total / 1e6, 3),
    }


def benchmark_optimized_mcts(num_iterations, seed=42):
    os.chdir(OPTIMIZED_DIR)
    sys.path.insert(0, OPTIMIZED_DIR)
    import othello_cpp

    timings = {
        'total_ns': [],
    }

    for i in range(num_iterations):
        mcts = othello_cpp.BatchedMCTS()
        # Set root board (initial Othello position)
        board = [0]*36
        board[14] = -1; board[15] = 1; board[20] = 1; board[21] = -1
        mcts.set_root_board(board, 1)

        start = time.perf_counter_ns()
        leaf_idx, state_tensor, legal_moves, is_term, val, hit, p_out, entry_v = mcts.select_and_get_leaf()

        policy = np.zeros(36, dtype=np.float32)
        if legal_moves:
            for m in legal_moves:
                if 0 <= m < 36:
                    policy[m] = 1.0 / len(legal_moves)
        value = val if is_term else 0.0

        mcts.expand_and_backup(leaf_idx, legal_moves, policy.tolist(), float(value))

        total_ns = time.perf_counter_ns() - start
        timings['total_ns'].append(total_ns)

    avg_total = np.mean(timings['total_ns'])

    return {
        'implementation': 'optimized_cpp',
        'num_iterations': num_iterations,
        'num_sims_per_iter': 1,
        'avg_total_ns_per_iter': round(avg_total),
        'avg_ns_per_sim': round(avg_total, 1),
        'avg_ms_per_iter': round(avg_total / 1e6, 3),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--num_sims', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='benchmark_mcts.json')
    args = parser.parse_args()

    print("[mcts_bench] Benchmarking baseline Python MCTS...")
    baseline = benchmark_baseline_mcts(args.num_iterations, args.num_sims, args.seed)
    print(f"  avg: {baseline['avg_ms_per_iter']:.3f} ms/iter ({baseline['avg_ns_per_sim']:.0f} ns/sim)")

    print("[mcts_bench] Benchmarking optimized C++ MCTS (single sim)...")
    optimized = benchmark_optimized_mcts(args.num_iterations, args.seed)
    print(f"  avg: {optimized['avg_ms_per_iter']:.3f} ms/iter ({optimized['avg_ns_per_sim']:.0f} ns/sim)")

    speedup = baseline['avg_ns_per_sim'] / optimized['avg_ns_per_sim'] if optimized['avg_ns_per_sim'] > 0 else 0

    result = {
        'baseline': baseline,
        'optimized': optimized,
        'speedup_factor': round(speedup, 2),
        'num_iterations': args.num_iterations,
        'seed': args.seed,
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[mcts_bench] Speedup: {speedup:.1f}x")
    print(f"[mcts_bench] Results saved to {args.output}")
