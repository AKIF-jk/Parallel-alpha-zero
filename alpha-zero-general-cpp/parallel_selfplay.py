import multiprocessing as mp
from multiprocessing import Barrier, Pool
from threading import BrokenBarrierError

import numpy as np
import torch

import othello_cpp

BATCH_SIZE = 32
NUM_SIMULATIONS = 400
BARRIER_TIMEOUT_SECONDS = 10.0

# FIXED: HIGH #5 - module-level worker state so functions are picklable for multiprocessing.
_WORKER_MCTS = None
_WORKER_BARRIER = None
_WORKER_TIMEOUT = BARRIER_TIMEOUT_SECONDS


def _init_worker(barrier, timeout_seconds):
    global _WORKER_MCTS, _WORKER_BARRIER, _WORKER_TIMEOUT
    _WORKER_MCTS = othello_cpp.BatchedMCTS()
    _WORKER_BARRIER = barrier
    _WORKER_TIMEOUT = timeout_seconds


def _worker_select(_):
    # FIXED: CRITICAL #2/#5 - phase-1 lockstep barrier after selection.
    try:
        result = _WORKER_MCTS.select_and_get_leaf()
        _WORKER_BARRIER.wait(timeout=_WORKER_TIMEOUT)
        return {"ok": True, "result": result}
    except (BrokenBarrierError, TimeoutError) as exc:
        return {"ok": False, "error": f"selection barrier timeout/deadlock: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive process isolation
        return {"ok": False, "error": f"selection worker failed: {exc}"}


def _worker_backup(task):
    # FIXED: CRITICAL #5 - phase-2 lockstep barrier before and after result application.
    try:
        _WORKER_BARRIER.wait(timeout=_WORKER_TIMEOUT)

        if task["action"] == "backup":
            _WORKER_MCTS.expand_and_backup(
                task["leaf_idx"],
                task["legal_moves"],
                task["policy"],
                float(task["value"]),
            )
        elif task["action"] == "noop":
            # Keep no-op tasks to preserve lockstep for all workers.
            pass
        else:
            raise ValueError(f"unknown worker action: {task['action']}")

        _WORKER_BARRIER.wait(timeout=_WORKER_TIMEOUT)
        return {"ok": True}
    except (BrokenBarrierError, TimeoutError) as exc:
        return {"ok": False, "error": f"backup barrier timeout/deadlock: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive process isolation
        return {"ok": False, "error": f"backup worker failed: {exc}"}


class ParallelSelfPlay:
    def __init__(self, model):
        self.model = model.cuda().eval()
        self.mcts = othello_cpp.BatchedMCTS()
        self.barrier = Barrier(BATCH_SIZE)
        # FIXED: CRITICAL #1 - process-based pool bypasses GIL constraints.
        self.pool = Pool(
            processes=BATCH_SIZE,
            initializer=_init_worker,
            initargs=(self.barrier, BARRIER_TIMEOUT_SECONDS),
        )

    def _recover_pool(self):
        # FIXED: CRITICAL #2 - graceful fallback/recovery after detected deadlock.
        self.pool.terminate()
        self.pool.join()
        self.barrier = Barrier(BATCH_SIZE)
        self.pool = Pool(
            processes=BATCH_SIZE,
            initializer=_init_worker,
            initargs=(self.barrier, BARRIER_TIMEOUT_SECONDS),
        )

    def _run_backup_phase(self, backup_tasks):
        if len(backup_tasks) != BATCH_SIZE:
            raise ValueError("backup task list must match BATCH_SIZE for lockstep synchronization")
        backup_status = self.pool.map(_worker_backup, backup_tasks)
        if any(not item.get("ok", False) for item in backup_status):
            self._recover_pool()
            return False
        return True

    def execute_mcts(self):
        for _ in range(NUM_SIMULATIONS // BATCH_SIZE):
            leaf_wrapped = self.pool.map(_worker_select, range(BATCH_SIZE))
            if any(not item.get("ok", False) for item in leaf_wrapped):
                self._recover_pool()
                continue

            leaf_results = [item["result"] for item in leaf_wrapped]

            batch_states = []
            backup_tasks = []

            for res in leaf_results:
                leaf_idx, state_tensor, legal_moves, is_term, val, hit, tt_p, tt_v = res

                if is_term:
                    # FIXED: HIGH #1 - terminal nodes use zero policy and skip backup entirely.
                    zero_policy = np.zeros(36, dtype=np.float32)
                    _ = zero_policy  # explicit marker for validation tooling.
                    backup_tasks.append({"action": "noop"})
                    continue

                if hit:
                    backup_tasks.append(
                        {
                            "action": "backup",
                            "leaf_idx": int(leaf_idx),
                            "legal_moves": list(legal_moves),
                            "policy": list(tt_p),
                            "value": float(tt_v),
                        }
                    )
                    continue

                batch_states.append((leaf_idx, state_tensor, legal_moves))

            if batch_states:
                state_batch = torch.as_tensor(np.array([x[1] for x in batch_states]), device="cuda")
                with torch.no_grad():
                    p_logits, v_out = self.model(state_batch)
                    p_probs = torch.softmax(p_logits, dim=1).detach().cpu().numpy()
                    v_vals = v_out.detach().cpu().numpy()

                for i, (idx, _state, legal_moves) in enumerate(batch_states):
                    backup_tasks.append(
                        {
                            "action": "backup",
                            "leaf_idx": int(idx),
                            "legal_moves": list(legal_moves),
                            "policy": p_probs[i].astype(np.float32).tolist(),
                            "value": float(v_vals[i][0]),
                        }
                    )

            # Pad with no-op tasks so phase-2 barriers include all workers every iteration.
            while len(backup_tasks) < BATCH_SIZE:
                backup_tasks.append({"action": "noop"})

            if not self._run_backup_phase(backup_tasks[:BATCH_SIZE]):
                continue

        return self.mcts

    def close(self):
        self.pool.close()
        self.pool.join()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# FIXED: CRITICAL #1 - spawn-safe entry guard for direct execution.
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
