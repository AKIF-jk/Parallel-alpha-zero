import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/alphazero_project/checkpoints')
args_cli = parser.parse_args()

CHECKPOINT_DIR = args_cli.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import logging
import torch
from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn, args as nnet_args
from utils import dotdict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

log.info('CUDA available: %s', torch.cuda.is_available())
if torch.cuda.is_available():
    log.info('CUDA device: %s', torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

# Technique C runs on Colab T4, so use a larger GPU-friendly training setup.
nnet_args.epochs = 15
nnet_args.batch_size = 128

# Reset profiler metrics
import profiler
profiler.iteration_metrics_list = []
profiler.gpu_utilization_list = []
profiler.mcts_sims_per_sec_list = []
profiler.peak_ram_list = []
profiler.cache_hit_rate_per_iter = []
profiler.gpu_calls_per_iter = []
profiler.avg_gpu_batch_size = []
profiler.win_rate_vs_greedy = 0.0

args = dotdict({
    'numIters': 5,
    'numEps': 48,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 35,
    'arenaCompare': 40,
    'greedyCompare': 40,
    'cpuct': 1,
    'checkpoint': CHECKPOINT_DIR,
    'load_model': False,
    'load_folder_file': (CHECKPOINT_DIR, 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    log.info('Starting Technique C training (lockstep batched self-play)')
    c.learn()

    metrics = profiler.save_metrics(CHECKPOINT_DIR, filename="technique_c_metrics.json")
    log.info('Technique C metrics saved: %s', metrics)


if __name__ == "__main__":
    main()
