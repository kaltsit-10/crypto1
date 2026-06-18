"""Massively Parallel High-BS Pump Solver for P3 SIS (384-core optimized).

Strategy: G6K pump is sequential per-block, but the sieve inside each pump call
uses multi-threading (OpenMP). To fully utilize 384 cores:
  1. Spawn N independent worker processes (each with its own G6K instance)
  2. Each worker runs focused pump on top blocks with its own random seed
  3. Different seeds → different BKZ evolution paths → better exploration
  4. Master collects results, auto-stops when any worker finds linf ≤ 16

Thread allocation:
  - With 384 physical cores, we can run many parallel workers
  - Each worker uses threads=8 for internal sieve parallelism
  - Target: 40 workers × 8 threads = 320 cores for sieve + 64 for overhead/LLL
  - Workers are independent → no GIL contention, no shared state

Key optimization: Workers don't need to finish full β sequence.
They check linf after each pump call and broadcast if improved.
"""
import sys, os, time, traceback, json
import numpy as np
import ast
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
import signal

# === Globals (set per-worker) ===
BASE = os.path.dirname(os.path.abspath(__file__))
P3_FILE = os.path.join(BASE, "crypto1_repo", "第三题", "problem3.txt")
RESDIR = os.path.join(BASE, "crypto1_repo", "第三题", "results_p3")
os.makedirs(RESDIR, exist_ok=True)

with open(P3_FILE) as f:
    A_GLOBAL = np.array(ast.literal_eval(
        f.read().strip().split('\n')[0].split('=', 1)[1].strip()
    ), dtype=np.int64)

M, N = A_GLOBAL.shape
Q_GLOBAL = int(np.max(A_GLOBAL)) + 1
DIM = M + N
GAMMA = 16


def centered_u(u_raw):
    u = u_raw % Q_GLOBAL
    return np.where(u >= Q_GLOBAL // 2, u - Q_GLOBAL, u).astype(np.int64)


def true_linf(vec):
    v = vec[:M].astype(np.int64)
    u = centered_u(vec[M:].astype(np.int64))
    if np.all(v == 0) and np.all(u == 0):
        return 999
    return max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))


def best_row_info(B):
    best_linf, best_idx = 999, -1
    for i in range(DIM):
        vec = np.array([int(B[i, j]) for j in range(DIM)], dtype=np.int64)
        li = true_linf(vec)
        if li < best_linf:
            best_linf, best_idx = li, i
    return best_idx, best_linf


def mk_gso_np(B):
    from fpylll import IntegerMatrix, GSO
    m = IntegerMatrix(DIM, DIM)
    for i in range(DIM):
        for j in range(DIM):
            m[i, j] = int(B[i, j])
    mg = GSO.Mat(m, U=IntegerMatrix.identity(DIM),
                 UinvT=IntegerMatrix.identity(DIM))
    mg.update_gso()
    return mg


def load_basis(path):
    return np.load(path).astype(np.int64)


def extract_basis(g):
    Bn = np.zeros((DIM, DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM):
            Bn[i, j] = int(g.M.B[i, j])
    return Bn


def worker_process(worker_id, basis_path, result_queue, stop_event,
                   threads, beta_start, beta_end, max_kappa):
    """Single worker: load basis, run focused pump, report results.

    Each worker gets a unique worker_id which seeds its random exploration.
    Workers operate INDEPENDENTLY with no inter-worker communication.
    """
    # Suppress G6K warnings in workers
    import warnings
    warnings.filterwarnings("ignore")

    from g6k import Siever, SieverParams
    from g6k.algorithms.pump import pump
    from g6k.utils.stats import dummy_tracer

    try:
        B = load_basis(basis_path)
        _, start_linf = best_row_info(B)

        # Build beta sequence
        betas = list(range(beta_start, beta_end + 1, 5))

        best_linf = start_linf
        best_vec = None
        best_beta = 0

        # Each worker uses worker_id to seed its random walk
        # Seed is embedded in Siever creation via seed parameter
        base_seed = worker_id * 100000

        for beta in betas:
            if stop_event.is_set():
                break

            dim4free = int(11.5 + 0.075 * beta)

            for kappa in range(0, max_kappa + 1):
                if stop_event.is_set():
                    break
                if kappa + beta > DIM:
                    break

                actual_beta = min(beta, DIM - kappa)
                lost_dim = beta - actual_beta
                actual_d4f = max(dim4free - lost_dim, 0)

                g = None
                for retry in range(2):
                    try:
                        seed_val = base_seed + beta * 1000 + kappa * 10 + retry
                        p = SieverParams()
                        p._set("threads", threads)
                        p._set("saturation_ratio", 0.3)
                        p._set("db_size_factor", 3.0)

                        g = Siever(mk_gso_np(B), p, seed=seed_val)
                        pump(g, dummy_tracer, kappa, actual_beta, actual_d4f,
                             down_sieve=False)

                        try:
                            g.lll(0, DIM)
                        except Exception:
                            pass
                        break
                    except Exception:
                        if retry == 1:
                            g = None

                if g is None:
                    continue

                B = extract_basis(g)
                ri, li = best_row_info(B)

                if li < best_linf:
                    best_linf = li
                    best_vec = B[ri].copy()
                    best_beta = beta

                    # Report improvement to master
                    v = best_vec[:M].astype(np.int64)
                    u = centered_u(best_vec[M:].astype(np.int64))
                    lv = int(np.max(np.abs(v)))
                    lu = int(np.max(np.abs(u)))

                    result_queue.put({
                        'worker': worker_id,
                        'linf': best_linf,
                        'lv': lv,
                        'lu': lu,
                        'beta': beta,
                        'kappa': kappa,
                        'basis': B.copy(),
                        'vec': np.concatenate([v, u])
                    })

                    if best_linf <= GAMMA:
                        stop_event.set()
                        break

            # Save intermediate basis per worker
            if beta >= 100 and beta % 10 == 0:
                fpath = os.path.join(RESDIR, f"worker{worker_id}_b{beta}_l{best_linf}.npy")
                np.save(fpath, B)

        # Final report
        result_queue.put({
            'worker': worker_id,
            'linf': best_linf,
            'beta': best_beta,
            'done': True,
            'basis': B.copy()
        })

    except Exception as e:
        result_queue.put({
            'worker': worker_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def master_loop(basis_path, n_workers=40, threads=8,
                beta_start=90, beta_end=130, max_kappa=5):
    """Master process: spawn workers, collect results, stop when solved."""

    print(f"{'='*70}")
    print(f"Parallel Pump Solver: {n_workers} workers × {threads} threads each")
    print(f"β range: {beta_start}→{beta_end}, top kappa: 0..{max_kappa}")
    print(f"Total parallel pump contexts: ~{n_workers * (beta_end-beta_start)//5 * (max_kappa+1)}")
    print(f"{'='*70}\n")

    # Load and check starting basis
    B0 = load_basis(basis_path)
    _, start_linf = best_row_info(B0)
    print(f"Starting basis: {os.path.basename(basis_path)}, linf={start_linf}")

    # Shared state
    result_queue = Queue()
    stop_event = Event()

    # Spawn workers
    workers = []
    t_start = time.time()

    for wid in range(n_workers):
        p = Process(target=worker_process, args=(
            wid, basis_path, result_queue, stop_event,
            threads, beta_start, beta_end, max_kappa
        ))
        p.start()
        workers.append(p)

    print(f"Launched {n_workers} workers. Monitoring...\n")

    # Monitor and collect results
    best_overall = start_linf
    best_result = None
    active_workers = n_workers
    completed_workers = 0
    improvements = []
    last_report = time.time()

    try:
        while active_workers > 0 and not stop_event.is_set():
            try:
                result = result_queue.get(timeout=5)

                if 'error' in result:
                    print(f"  [W{result['worker']}] ERROR: {result['error']}")
                    completed_workers += 1
                    active_workers -= 1
                    continue

                if result.get('done'):
                    completed_workers += 1
                    active_workers -= 1
                    print(f"  [W{result['worker']}] DONE: best linf={result['linf']} "
                          f"({completed_workers}/{n_workers} finished)")
                    if result['linf'] < best_overall:
                        best_overall = result['linf']
                        best_result = result
                    continue

                # Improvement event
                li = result['linf']
                if li <= best_overall:
                    best_overall = li
                    best_result = result
                    improvements.append(result)

                    dt = time.time() - t_start
                    tag = " *** SOLVED ***" if li <= GAMMA else " *** NEW BEST ***"
                    print(f"\n{'='*60}")
                    print(f"[W{result['worker']}] β={result['beta']}, κ={result['kappa']}: "
                          f"linf={li} (lv={result['lv']}, lu={result['lu']}) "
                          f"time={dt:.0f}s{tag}")
                    print(f"{'='*60}\n")

                    # Save best basis
                    np.save(os.path.join(RESDIR, "parallel_best.npy"), result['basis'])
                    np.save(os.path.join(RESDIR, "parallel_best_vec.npy"), result['vec'])

                    if li <= GAMMA:
                        stop_event.set()
                        print("\n!!! SOLVED !!! Stopping all workers...")
                        break

                # Periodic status
                if time.time() - last_report > 30:
                    dt = time.time() - t_start
                    print(f"  [status {dt:.0f}s] active={active_workers} "
                          f"completed={completed_workers} "
                          f"best={best_overall} "
                          f"improvements={len(improvements)}")
                    last_report = time.time()

            except Exception:
                # Queue timeout, check if workers are still alive
                for w in workers:
                    if not w.is_alive() and w.exitcode is not None:
                        if w.pid is not None:  # worker was started
                            completed_workers += 1
                            active_workers -= 1
                if active_workers <= 0:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping workers...")
        stop_event.set()

    # Cleanup
    stop_event.set()
    for w in workers:
        if w.is_alive():
            w.terminate()
            w.join(timeout=5)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Final Results: {elapsed:.0f}s")
    print(f"  Best linf: {best_overall}")
    print(f"  Total improvements: {len(improvements)}")
    if best_result:
        print(f"  From worker {best_result['worker']} at β={best_result['beta']}")
    print(f"  Workers completed: {completed_workers}/{n_workers}")
    print(f"{'='*70}")

    return best_overall, best_result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", type=str, default=None)
    ap.add_argument("--workers", type=int, default=32,
                    help="Number of parallel worker processes")
    ap.add_argument("--threads", type=int, default=8,
                    help="Threads per worker (for internal sieve)")
    ap.add_argument("--beta-start", type=int, default=90)
    ap.add_argument("--beta-end", type=int, default=130)
    ap.add_argument("--max-kappa", type=int, default=5,
                    help="Top blocks to process (0..max_kappa)")
    args = ap.parse_args()

    # Use 'fork' for faster startup (default on Linux)
    mp.set_start_method('fork', force=True)

    if args.basis:
        bpath = args.basis
    else:
        bpath = os.path.join(RESDIR, "g6k_push_b95_p3.npy")

    if not os.path.exists(bpath):
        print(f"Basis not found: {bpath}")
        sys.exit(1)

    master_loop(
        bpath,
        n_workers=args.workers,
        threads=args.threads,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        max_kappa=args.max_kappa,
    )
