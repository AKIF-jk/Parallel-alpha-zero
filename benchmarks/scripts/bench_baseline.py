#!/usr/bin/env python3
"""
Runs self-play on BASELINE Python implementation and measures throughput.
Outputs: games completed, time elapsed, GPS, per-iteration breakdown.
"""
import sys
import time
import os
import json
import argparse

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-master'))
sys.path.insert(0, BASELINE_DIR)
os.chdir(BASELINE_DIR)

import numpy as np
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from Arena import Arena

def benchmark_self_play(num_games, num_mcts_sims, num_iters=1, seed=42):
    np.random.seed(seed)
    game = OthelloGame(6)
    nnet = NNetWrapper(game)

    args = type('obj', (object,), {
        'cpuct': 1.0,
        'numMCTSSims': num_mcts_sims,
    })()

    def mcts_player(board):
        mcts = MCTS(game, nnet, args)
        probs = mcts.getActionProb(board, temp=1.0)
        return np.argmax(probs)

    arena = Arena(mcts_player, mcts_player, game, display=lambda b: None)

    total_time = 0
    game_times = []

    for g in range(num_games):
        np.random.seed(seed + g)
        start = time.perf_counter()
        result = arena.playGame()
        elapsed = time.perf_counter() - start
        game_times.append(elapsed)
        total_time += elapsed

    gps = num_games / total_time if total_time > 0 else 0

    results = {
        'implementation': 'baseline_python',
        'num_games': num_games,
        'num_mcts_sims': num_mcts_sims,
        'total_time_sec': round(total_time, 3),
        'gps': round(gps, 3),
        'game_times': [round(t, 3) for t in game_times],
        'mean_game_time': round(np.mean(game_times), 3),
        'std_game_time': round(np.std(game_times), 3),
        'seed': seed,
    }
    return results


def benchmark_full_iteration(num_eps, num_mcts_sims, seed=42):
    np.random.seed(seed)
    game = OthelloGame(6)
    nnet = NNetWrapper(game)

    args = type('obj', (object,), {
        'numIters': 1,
        'numEps': num_eps,
        'numMCTSSims': num_mcts_sims,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'arenaCompare': 0,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
    })()

    from Coach import Coach

    c = Coach(game, nnet, args)

    start = time.perf_counter()
    c.learn()
    elapsed = time.perf_counter() - start

    results = {
        'implementation': 'baseline_python',
        'num_eps': num_eps,
        'num_mcts_sims': num_mcts_sims,
        'total_time_sec': round(elapsed, 3),
        'gps': round(num_eps / elapsed, 3),
        'seed': seed,
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['self_play', 'full_iteration'], default='self_play')
    parser.add_argument('--num_games', type=int, default=20)
    parser.add_argument('--num_eps', type=int, default=5)
    parser.add_argument('--num_mcts_sims', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='benchmark_baseline.json')
    args = parser.parse_args()

    print(f"[baseline] Running {args.mode} benchmark...")

    if args.mode == 'self_play':
        result = benchmark_self_play(args.num_games, args.num_mcts_sims, seed=args.seed)
    else:
        result = benchmark_full_iteration(args.num_eps, args.num_mcts_sims, seed=args.seed)

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[baseline] GPS: {result['gps']:.3f}")
    print(f"[baseline] Total time: {result['total_time_sec']:.1f}s")
    print(f"[baseline] Results saved to {args.output}")
