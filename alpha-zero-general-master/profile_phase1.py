import argparse
import cProfile
import pstats
import random
import time

import numpy as np

from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from utils import dotdict


class DummyNNet:
    """Fast deterministic network stub for CPU profiling."""

    def __init__(self, game):
        self.action_size = game.getActionSize()
        self._uniform = np.full(self.action_size, 1.0 / self.action_size, dtype=np.float32)

    def predict(self, _board):
        return self._uniform, 0.0


def sample_positions(game, count, max_depth, seed):
    rng = random.Random(seed)
    positions = []
    board = game.getInitBoard()
    player = 1

    while len(positions) < count:
        depth = rng.randint(0, max_depth)
        board = game.getInitBoard()
        player = 1

        for _ in range(depth):
            canonical = game.getCanonicalForm(board, player)
            valids = game.getValidMoves(canonical, 1)
            legal_actions = np.flatnonzero(valids)
            if len(legal_actions) == 0:
                break
            action = int(rng.choice(legal_actions))
            board, player = game.getNextState(board, player, action)
            if game.getGameEnded(board, player) != 0:
                break

        positions.append(game.getCanonicalForm(board, player))

    return positions


def run_profile(board_size, num_positions, sims_per_position, max_depth, seed, threads=1):
    game = OthelloGame(board_size, use_zobrist=False)
    nnet = DummyNNet(game)
    args = dotdict({
        "numMCTSSims": sims_per_position,
        "numMCTSThreads": int(threads),
        "cpuct": 1.0,
        "nnCacheMaxSize": 500000,
        "actionArrayPoolSize": 1024,
    })
    mcts = MCTS(game, nnet, args)
    positions = sample_positions(game, num_positions, max_depth, seed)

    started = time.perf_counter()
    for position in positions:
        mcts.getActionProb(position, temp=1)
    elapsed = time.perf_counter() - started

    return elapsed, len(positions) * sims_per_position, mcts.cache_stats()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 CPU profiling helper.")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stats-file", type=str, default="phase1_profile.stats")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    elapsed, total_sims, cache_stats = run_profile(
        board_size=args.board_size,
        num_positions=args.positions,
        sims_per_position=args.sims,
        max_depth=args.max_depth,
        seed=args.seed,
        threads=args.threads,
    )
    profiler.disable()
    profiler.dump_stats(args.stats_file)

    sims_per_sec = total_sims / elapsed if elapsed > 0 else 0.0
    print(f"profile saved: {args.stats_file}")
    print(f"positions={args.positions} sims={total_sims} elapsed={elapsed:.3f}s sims_per_sec={sims_per_sec:.1f}")
    print(
        "cache: hits={hits} misses={misses} hit_rate={hit_rate_pct:.2f}% size={cache_size}".format(
            **cache_stats
        )
    )

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
