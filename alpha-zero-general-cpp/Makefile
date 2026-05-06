VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: all train test clean rebuild venv pit

# Default target
all: venv cpp

# Create virtual environment and install dependencies
venv:
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install numpy coloredlogs tqdm pybind11
	touch $(VENV)

# Build C++ extension
cpp:
	$(PYTHON) setup.py build_ext --inplace

# Run training with reduced params
train: venv cpp
	$(PYTHON) main.py

# Run pit.py to play against trained model
pit: venv cpp
	$(PYTHON) pit.py

# Quick test run (1 iteration, 5 episodes)
test: venv cpp
	@echo "Running quick test (1 iter, 5 eps)..."
	$(PYTHON) -c "\
import logging; \
import coloredlogs; \
coloredlogs.install(level='INFO'); \
from Coach import Coach; \
from othello.OthelloGame import OthelloGame as Game; \
from othello.pytorch.NNet import NNetWrapper as nn; \
from utils import dotdict; \
args = dotdict({'numIters': 1, 'numEps': 5, 'tempThreshold': 15, 'updateThreshold': 0.6, 'maxlenOfQueue': 200000, 'numMCTSSims': 10, 'arenaCompare': 5, 'cpuct': 1, 'checkpoint': './temp/', 'load_model': False, 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'), 'numItersForTrainExamplesHistory': 20}); \
g = Game(6); \
nnet = nn(g); \
c = Coach(g, nnet, args); \
c.learn(); \
print('Test completed successfully!')"

# Clean build artifacts
clean:
	rm -rf build/ *.so __pycache__/
	rm -rf othello/__pycache__/ othello/cpp/*.o
	find . -name "*.pyc" -delete

# Rebuild everything from scratch
rebuild: clean
	$(MAKE) cpp

# Test C++ module only
test-cpp: venv cpp
	$(PYTHON) -c "import othello_cpp; m = othello_cpp.BatchedMCTS(); print('C++ module OK')"
