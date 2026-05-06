#!/usr/bin/env python3
"""
Scaling benchmark: measures throughput with varying worker counts.
Only tests the optimized implementation (baseline is single-threaded).
"""
import sys
import time
import os
import json
import argparse
import multiprocessing as mp

OPTIMIZED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-cpp'))
sys.path.insert(0, OPTIMIZED_DIR)
os.chdir(OPTIMIZED_DIR)

import numpy as np
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper
from Arena import Arena


def run_game(num_mcts_sims, seed):
    """Run a single game and return elapsed time."""
    np.random.seed(seed)
    game = OthelloGame(6)
    nnet = NNetWrapper(game)

    args = type('obj', (object,), {
        'cpuct': 1.0,
        'numMCTSSims': num_mcts_sims,
    })()

    from MCTS import MCTS

    def player1(board):
        mcts = MCTS(game, nnet, args)
        probs = mcts.getActionProb(board, temp=1.0)
        return np.argmax(probs)

    def player2(board):
        mcts = MCTS(game, nnet, args)
        probs = mcts.getActionProb(board, temp=1.0)
        return np.argmax(probs)

    arena = Arena(player1, player2, game, display=lambda b: None)
    start = time.perf_counter()
    result = arena.playGame()
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_scaling(num_games, num_mcts_sims, workers_list, seed=42):
    results = {}

    for num_workers in workers_list:
        print(f"[scaling] Testing {num_workers} worker(s)...")

        seeds = [seed + i for i in range(num_games)]

        start = time.perf_counter()

        if num_workers == 1:
            for s in seeds:
                run_game(num_mcts_sims, s)
        else:
            with mp.Pool(processes=num_workers) as pool:
                pool.starmap(run_game, [(num_mcts_sims, s) for s in seeds])

        total_time = time.perf_counter() - start

        gps = num_games / total_time if total_time > 0 else 0

        results[num_workers] = {
            'num_workers': num_workers,
            'num_games': num_games,
            'total_time_sec': round(total_time, 3),
            'gps': round(gps, 3),
        }

        print(f"  time={total_time:.1f}s, GPS={gps:.3f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', type=int, default=10)
    parser.add_argument('--num_mcts_sims', type=int, default=25)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='benchmark_scaling.json')
    args = parser.parse_args()

    print(f"[scaling_bench] Running scaling benchmark: {args.workers} workers, {args.num_games} games...")
    results = benchmark_scaling(args.num_games, args.num_mcts_sims, args.workers, args.seed)

    output = {
        'num_games': args.num_games,
        'num_mcts_sims': args.num_mcts_sims,
        'seed': args.seed,
        'results': results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[scaling_bench] Results saved to {args.output}")
