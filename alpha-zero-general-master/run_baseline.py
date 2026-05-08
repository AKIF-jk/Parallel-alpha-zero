import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/alphazero_project/checkpoints',
                    help='Checkpoint directory path')
args_cli = parser.parse_args()

CHECKPOINT_DIR = args_cli.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import logging
from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

args = dotdict({
    'numIters': 5,
    'numEps': 20,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'numMCTSThreads': 1,
    'arenaCompare': 20,
    'cpuct': 1,
    'nnCacheMaxSize': 500000,
    'terminalCacheMaxSize': 250000,
    'inferenceCacheMaxSize': 500000,
    'actionArrayPoolSize': 512,
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

    log.info('Starting the learning process')
    c.learn()

    # Save metrics
    import profiler
    metrics = profiler.save_metrics(CHECKPOINT_DIR, filename="baseline_metrics.json")
    log.info('Baseline metrics saved: %s', metrics)

if __name__ == "__main__":
    main()
