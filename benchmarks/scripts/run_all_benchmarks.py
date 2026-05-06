#!/usr/bin/env python3
"""
Master benchmark runner: executes all experiments and collects results.
Usage: python3 run_all_benchmarks.py [--quick]
"""
import sys
import os
import json
import time
import subprocess

SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_script(script_name, args_dict, timeout=600, use_venv='optimized'):
    """Run a benchmark script and return its output file path."""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if use_venv == 'baseline':
        python_path = os.path.join(base_dir, 'alpha-zero-general-master', 'venv', 'bin', 'python3')
    else:
        python_path = os.path.join(base_dir, 'alpha-zero-general-cpp', 'venv', 'bin', 'python3')

    cmd = [python_path, script_path]
    for key, value in args_dict.items():
        if isinstance(value, list):
            cmd.append(f'--{key}')
            for v in value:
                cmd.append(str(v))
        elif isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))

    output_file = args_dict.get('output', 'output.json')
    output_path = os.path.join(RESULTS_DIR, output_file)
    cmd[cmd.index('--output') + 1] = output_path

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr}")
        if result.returncode != 0:
            print(f"[ERROR] Script exited with code {result.returncode}")
            return None
        print(f"Completed in {elapsed:.1f}s")
        return output_path
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Script exceeded {timeout}s")
        return None


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run fewer games/iterations for faster results')
    parser.add_argument('--skip-scaling', action='store_true', help='Skip scaling benchmark (takes long)')
    parser.add_argument('--skip-tournament', action='store_true', help='Skip correctness tournament')
    args = parser.parse_args()

    num_games = 10 if args.quick else 20
    num_eps = 2 if args.quick else 5
    num_mcts_sims = 15 if args.quick else 25
    mcts_iters = 20 if args.quick else 100
    tournament_games = 10 if args.quick else 20

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AlphaZero Benchmark Suite - Real Measurements     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Games: {num_games}, Sims: {num_mcts_sims}")

    results = {}

    # Experiment 1: Baseline self-play throughput
    print("\n📊 Experiment 1: Baseline Throughput")
    path = run_script('bench_baseline.py', {
        'mode': 'self_play',
        'num_games': num_games,
        'num_mcts_sims': num_mcts_sims,
        'seed': 42,
        'output': 'baseline_selfplay.json',
    }, use_venv='baseline')
    results['baseline_selfplay'] = load_json(path)

    # Experiment 2: Optimized self-play throughput
    print("\n📊 Experiment 2: Optimized Throughput")
    path = run_script('bench_optimized.py', {
        'mode': 'self_play',
        'num_games': num_games,
        'num_mcts_sims': num_mcts_sims,
        'seed': 42,
        'output': 'optimized_selfplay.json',
    }, use_venv='optimized')
    results['optimized_selfplay'] = load_json(path)

    # Experiment 3: MCTS microbenchmark
    print("\n📊 Experiment 3: MCTS Microbenchmark")
    path = run_script('bench_mcts_micro.py', {
        'num_iterations': mcts_iters,
        'num_sims': num_mcts_sims,
        'seed': 42,
        'output': 'mcts_micro.json',
    }, use_venv='optimized')
    results['mcts_micro'] = load_json(path)

    # Experiment 4: Scaling (optional, takes long)
    if not args.skip_scaling:
        print("\n📊 Experiment 4: Scaling Benchmark")
        path = run_script('bench_scaling.py', {
            'num_games': num_games // 2,
            'num_mcts_sims': num_mcts_sims,
            'workers': [1, 2, 4],
            'seed': 42,
            'output': 'scaling.json',
        }, use_venv='optimized')
        results['scaling'] = load_json(path)
    else:
        print("\n⏭️  Skipping scaling benchmark")

    # Experiment 5: Correctness tournament
    if not args.skip_tournament:
        print("\n📊 Experiment 5: Correctness Tournament")
        path = run_script('bench_tournament.py', {
            'num_games': tournament_games,
            'num_mcts_sims': num_mcts_sims,
            'seed': 42,
            'output': 'tournament.json',
        }, use_venv='optimized')
        results['tournament'] = load_json(path)
    else:
        print("\n⏭️  Skipping tournament")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")

    if results.get('baseline_selfplay'):
        b_gps = results['baseline_selfplay']['gps']
        print(f"Baseline GPS:     {b_gps:.3f}")

    if results.get('optimized_selfplay'):
        o_gps = results['optimized_selfplay']['gps']
        print(f"Optimized GPS:    {o_gps:.3f}")
        if results.get('baseline_selfplay'):
            b_gps = results['baseline_selfplay']['gps']
            print(f"Speedup:          {o_gps / b_gps:.1f}x")

    if results.get('mcts_micro'):
        speedup = results['mcts_micro']['speedup_factor']
        b_ns = results['mcts_micro']['baseline']['avg_ns_per_sim']
        o_ns = results['mcts_micro']['optimized']['avg_ns_per_sim']
        print(f"\nMCTS Speedup:     {speedup:.1f}x")
        print(f"Baseline:         {b_ns:.0f} ns/sim")
        print(f"Optimized:        {o_ns:.0f} ns/sim")

    if results.get('scaling'):
        print(f"\nScaling results:")
        for w, data in results['scaling'].get('results', {}).items():
            print(f"  W={w}: {data['gps']:.3f} GPS ({data['total_time_sec']:.1f}s)")

    if results.get('tournament'):
        t = results['tournament']
        print(f"\nTournament:")
        print(f"  Baseline wins:  {t['wins_baseline']} ({t['baseline_win_pct']}%)")
        print(f"  Optimized wins: {t['wins_optimized']} ({t['optimized_win_pct']}%)")
        print(f"  Draws:          {t['draws']} ({t['draw_pct']}%)")

    print(f"\nAll results saved to: {combined_path}")


if __name__ == '__main__':
    main()
