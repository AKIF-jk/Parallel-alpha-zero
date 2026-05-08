import argparse
import random
import time

import numpy as np

from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from utils import dotdict


class DummyNNet:
    """Deterministic stub network to isolate search/game-logic performance."""

    def __init__(self, game):
        self.action_size = game.getActionSize()
        self._uniform = np.full(self.action_size, 1.0 / self.action_size, dtype=np.float32)

    def predict(self, _board):
        return self._uniform, 0.0


def sample_positions(game, count, max_depth, seed):
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        board = game.getInitBoard()
        player = 1
        depth = rng.randint(0, max_depth)
        for _ in range(depth):
            canonical = game.getCanonicalForm(board, player)
            valids = game.getValidMoves(canonical, 1)
            legal = np.flatnonzero(valids)
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            board, player = game.getNextState(board, player, action)
            if game.getGameEnded(board, player) != 0:
                break
        positions.append(game.getCanonicalForm(board, player))
    return positions


def run_case(name, use_zobrist, use_bitboard, board_size, positions, sims, threads, max_depth, seed):
    game = OthelloGame(board_size, use_zobrist=use_zobrist, use_bitboard=use_bitboard)
    nnet = DummyNNet(game)
    args = dotdict({
        "numMCTSSims": sims,
        "numMCTSThreads": threads,
        "cpuct": 1.0,
        "nnCacheMaxSize": 500000,
        "actionArrayPoolSize": 1024,
    })
    mcts = MCTS(game, nnet, args)
    test_positions = sample_positions(game, positions, max_depth, seed)

    started = time.perf_counter()
    for pos in test_positions:
        mcts.getActionProb(pos, temp=1)
    elapsed = time.perf_counter() - started
    total_sims = positions * sims
    sps = total_sims / elapsed if elapsed > 0 else 0.0
    cache = mcts.cache_stats()
    return {
        "name": name,
        "elapsed": elapsed,
        "sims": total_sims,
        "sps": sps,
        "cache_hit_pct": cache["hit_rate_pct"],
        "cache_size": cache["cache_size"],
    }


def print_results(results):
    base = results[0]["sps"] if results else 0.0
    print()
    print(f"{'CASE':<20} {'SIMS/SEC':>10} {'ELAPSED(s)':>12} {'REL':>8} {'HIT%':>8} {'CACHE':>8}")
    print("-" * 72)
    for row in results:
        rel = (row["sps"] / base) if base > 0 else 0.0
        print(
            f"{row['name']:<20} "
            f"{row['sps']:>10.1f} "
            f"{row['elapsed']:>12.3f} "
            f"{rel:>8.2f} "
            f"{row['cache_hit_pct']:>8.2f} "
            f"{row['cache_size']:>8d}"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 3 backend benchmark (Python vs Zobrist vs Bitboard).")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.board_size != 8:
        print("Warning: bitboard backend is only available for 8x8; bitboard case may fall back to python.")

    configs = [
        ("python_tobytes", False, False),
        ("python_zobrist", True, False),
        ("bitboard_cpp", False, True),
    ]
    results = []
    for name, use_zobrist, use_bitboard in configs:
        try:
            row = run_case(
                name=name,
                use_zobrist=use_zobrist,
                use_bitboard=use_bitboard,
                board_size=args.board_size,
                positions=args.positions,
                sims=args.sims,
                threads=args.threads,
                max_depth=args.max_depth,
                seed=args.seed,
            )
            results.append(row)
        except Exception as exc:
            print(f"{name}: FAILED ({exc})")

    print_results(results)


if __name__ == "__main__":
    main()
