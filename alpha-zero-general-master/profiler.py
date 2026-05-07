import time
import subprocess
import threading
import resource
import json
import os

# Global metric storage
gpu_monitor = None
iteration_metrics_list = []
gpu_utilization_list = []
mcts_sims_per_sec_list = []
peak_ram_list = []
cache_hit_rate_per_iter = []
gpu_calls_per_iter = []
avg_gpu_batch_size = []
worker_utilization = []
examples_per_worker = []
win_rate_vs_greedy = 0.0
mcts_sim_count = 0
gpu_batch_boards = 0
gpu_batch_calls = 0
phase_start_times = {}

class GPUMonitor:
    def __init__(self):
        self.readings = []
        self.stop_event = threading.Event()
        self.thread = None
    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if res.returncode == 0:
                    self.readings.append(float(res.stdout.strip()))
            except Exception:
                pass
            time.sleep(2)
    def start(self):
        self.readings = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
    def average(self):
        return sum(self.readings)/len(self.readings) if self.readings else 0.0

def start_gpu_monitor():
    global gpu_monitor
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

def stop_gpu_monitor():
    global gpu_monitor, gpu_utilization_list
    if gpu_monitor:
        gpu_monitor.stop()
        gpu_utilization_list.append(gpu_monitor.average())
        gpu_monitor = None

def increment_mcts_sim():
    global mcts_sim_count
    mcts_sim_count += 1

def reset_mcts_sim():
    global mcts_sim_count
    mcts_sim_count = 0

def get_mcts_sim_count():
    global mcts_sim_count
    return mcts_sim_count

def reset_gpu_batch_stats():
    global gpu_batch_boards, gpu_batch_calls
    gpu_batch_boards = 0
    gpu_batch_calls = 0

def record_gpu_batch(batch_size):
    global gpu_batch_boards, gpu_batch_calls
    gpu_batch_boards += batch_size
    gpu_batch_calls += 1

def get_avg_gpu_batch_size():
    return gpu_batch_boards / gpu_batch_calls if gpu_batch_calls > 0 else 0.0

def get_gpu_batch_call_count():
    return gpu_batch_calls

def start_phase(name):
    phase_start_times[name] = time.perf_counter()

def end_phase(name):
    return time.perf_counter() - phase_start_times.get(name, time.perf_counter())

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB

def save_metrics(checkpoint_dir, filename="baseline_metrics.json"):
    global iteration_metrics_list, gpu_utilization_list, mcts_sims_per_sec_list, peak_ram_list
    global cache_hit_rate_per_iter, gpu_calls_per_iter, avg_gpu_batch_size
    global worker_utilization, examples_per_worker, win_rate_vs_greedy
    metrics = {
        "iteration_times": iteration_metrics_list,
        "gpu_utilization_pct": gpu_utilization_list,
        "mcts_sims_per_sec": mcts_sims_per_sec_list,
        "peak_ram_mb": peak_ram_list,
        "cache_hit_rate_per_iter": cache_hit_rate_per_iter,
        "gpu_calls_per_iter": gpu_calls_per_iter,
        "avg_gpu_batch_size": avg_gpu_batch_size,
        "worker_utilization": worker_utilization,
        "examples_per_worker": examples_per_worker,
        "win_rate_vs_greedy": win_rate_vs_greedy
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, filename), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
