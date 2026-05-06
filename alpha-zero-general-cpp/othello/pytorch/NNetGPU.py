import os
import sys
import time
import threading
import queue

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'gpu_thread_count': 4,
})


class NNetGPUWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        self.use_gpu = args.cuda and torch.cuda.is_available()
        
        if self.use_gpu:
            self.nnet.cuda()
            self.nnet = nn.DataParallel(self.nnet)
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        self.eval_mode = False
        self._batch_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._batch_thread = None
        self._start_batch_processor()

    def _start_batch_processor(self):
        def batch_processor():
            while True:
                batch_data = self._batch_queue.get()
                if batch_data is None:
                    break
                
                boards, result_queue = batch_data
                boards_tensor = torch.FloatTensor(boards).cuda() if self.use_gpu else torch.FloatTensor(boards)
                
                with torch.no_grad():
                    pi, v = self.nnet(boards_tensor)
                
                pi_np = torch.exp(pi).cpu().numpy()
                v_np = v.cpu().numpy()
                
                result_queue.put((pi_np, v_np))
        
        if self.use_gpu:
            self._batch_thread = threading.Thread(target=batch_processor, daemon=True)
            self._batch_thread.start()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.use_gpu:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        start = time.time()
        
        board = torch.FloatTensor(board.astype(np.float64))
        if self.use_gpu:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    def predict_batch(self, boards):
        if not self.use_gpu or len(boards) == 0:
            return self._predict_batch_cpu(boards)
        
        self.nnet.eval()
        boards_tensor = torch.FloatTensor(boards).cuda()
        
        with torch.no_grad():
            pis, vs = self.nnet(boards_tensor)
        
        return torch.exp(pis).cpu().numpy(), vs.cpu().numpy()

    def _predict_batch_cpu(self, boards):
        if len(boards) == 0:
            return np.array([]), np.array([])
        
        self.nnet.eval()
        boards_tensor = torch.FloatTensor(boards)
        
        with torch.no_grad():
            pis, vs = self.nnet(boards_tensor)
        
        return torch.exp(pis).numpy(), vs.numpy()

    def predict_async(self, board):
        result_queue = queue.Queue()
        board_np = board.astype(np.float64).copy()
        self._batch_queue.put((board_np[np.newaxis, :], result_queue))
        return result_queue.get()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        
        if self.use_gpu:
            torch.save({
                'state_dict': self.nnet.module.state_dict() if isinstance(self.nnet, nn.DataParallel) else self.nnet.state_dict(),
            }, filepath)
        else:
            torch.save({
                'state_dict': self.nnet.state_dict(),
            }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        
        map_location = 'cuda:0' if self.use_gpu else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        
        state_dict = checkpoint['state_dict']
        
        if isinstance(self.nnet, nn.DataParallel):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k] = v
                else:
                    new_state_dict['module.' + k] = v
            self.nnet.load_state_dict(new_state_dict)
        else:
            self.nnet.load_state_dict(state_dict)