#!/usr/bin/env python3
"""
Correctness tournament: Baseline vs C++ MCTS Optimized.
Plays games alternating which implementation starts.
"""
import sys
import os
import json
import argparse
import numpy as np

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-master'))
OPTIMIZED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alpha-zero-general-cpp'))


def run_tournament(num_games, num_mcts_sims, seed=42):
    # Baseline setup
    os.chdir(BASELINE_DIR)
    sys.path.insert(0, BASELINE_DIR)
    from othello.OthelloGame import OthelloGame
    from othello.pytorch.NNet import NNetWrapper
    from MCTS import MCTS as MCTS_Python
    from Arena import Arena

    # Optimized setup
    os.chdir(OPTIMIZED_DIR)
    sys.path.insert(0, OPTIMIZED_DIR)
    from MCTS_CPP import MCTS_CPP

    np.random.seed(seed)
    game = OthelloGame(6)
    nnet_baseline = NNetWrapper(game)
    nnet_optimized = NNetWrapper(game)

    args = type('obj', (object,), {
        'cpuct': 1.0,
        'numMCTSSims': num_mcts_sims,
    })()

    def baseline_player(board):
        mcts = MCTS_Python(game, nnet_baseline, args)
        probs = mcts.getActionProb(board, temp=1.0)
        return np.argmax(probs)

    def optimized_player(board):
        mcts = MCTS_CPP(game, nnet_optimized, args)
        probs = mcts.getActionProb(board, temp=1.0)
        return np.argmax(probs)

    wins_baseline = 0
    wins_optimized = 0
    draws = 0

    for g in range(num_games):
        np.random.seed(seed + g)

        if g % 2 == 0:
            arena = Arena(baseline_player, optimized_player, game, display=None)
        else:
            arena = Arena(optimized_player, baseline_player, game, display=None)

        game_result = arena.playGame()

        if game_result == 1:
            wins_baseline += 1
        elif game_result == -1:
            wins_optimized += 1
        else:
            draws += 1

        print(f"  Game {g+1}/{num_games}: result={game_result}", end='\r')

    print()
    results = {
        'num_games': num_games,
        'num_mcts_sims': num_mcts_sims,
        'wins_baseline': wins_baseline,
        'wins_optimized': wins_optimized,
        'draws': draws,
        'baseline_win_pct': round(wins_baseline / num_games * 100, 1),
        'optimized_win_pct': round(wins_optimized / num_games * 100, 1),
        'draw_pct': round(draws / num_games * 100, 1),
        'seed': seed,
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', type=int, default=10)
    parser.add_argument('--num_mcts_sims', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='benchmark_tournament.json')
    args = parser.parse_args()

    print(f"[tournament] Running {args.num_games} games (Baseline vs C++ MCTS)...")
    results = run_tournament(args.num_games, args.num_mcts_sims, args.seed)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[tournament] Baseline wins: {results['wins_baseline']} ({results['baseline_win_pct']}%)")
    print(f"[tournament] Optimized wins: {results['wins_optimized']} ({results['optimized_win_pct']}%)")
    print(f"[tournament] Draws: {results['draws']} ({results['draw_pct']}%)")
    print(f"[tournament] Results saved to {args.output}")