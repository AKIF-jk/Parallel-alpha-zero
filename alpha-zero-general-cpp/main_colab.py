import logging
import os

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNetGPU import NNetGPUWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 30,
    'numEps': 100,
    'tempThreshold': 30,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 300000,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    'cpuct': 1.5,

    'checkpoint': './checkpoint/',
    'load_model': False,
    'load_folder_file': ('./checkpoint','best.pth.tar'),
    'numItersForTrainExamplesHistory': 25,

    'gpu': True,
    'batch_size': 512,
    'lr': 0.001,
})


def setup_google_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(args.checkpoint, exist_ok=True)
        log.info("Google Drive mounted at %s", args.checkpoint)
    except Exception:
        log.warning("Not running in Google Colab - using local checkpoint directory")
        args.checkpoint = './checkpoint/'
        os.makedirs(args.checkpoint, exist_ok=True)


def main():
    import torch
    log.info('PyTorch version: %s', torch.__version__)
    log.info('CUDA available: %s', torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info('GPU: %s', torch.cuda.get_device_name(0))
    
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn()


if __name__ == "__main__":
    setup_google_drive()
    main()