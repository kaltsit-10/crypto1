"""Focused High-BS Pump: Only process top blocks (kappa=0..4) where the best row lives.
Strategy: Instead of full pump_n_jump_bkz_tour (178 blocks), only pump the top few
blocks that affect row 4 (linf=18). Each pump call uses high β (80→130).

With 384 cores, each pump call can use threads=32 for faster sieving.
"""
import sys, os, time, traceback
import numpy as np
import ast

BASE = os.path.dirname(os.path.abspath(__file__))
P3_FILE = os.path.join(BASE, "crypto1_repo", "第三题", "problem3.txt")
RESDIR = os.path.join(BASE, "crypto1_repo", "第三题", "results_p3")
os.makedirs(RESDIR, exist_ok=True)

with open(P3_FILE) as f:
    A = np.array(ast.literal_eval(
        f.read().strip().split('\n')[0].split('=', 1)[1].strip()
    ), dtype=np.int64)

M, N = A.shape
Q = int(np.max(A)) + 1
DIM = M + N
GAMMA = 16

print(f"P3: dim={DIM}, q={Q}, gamma={GAMMA}")

from fpylll import IntegerMatrix, GSO, LLL
from g6k import Siever, SieverParams
from g6k.algorithms.pump import pump
from g6k.utils.stats import dummy_tracer


def centered_u(u_raw):
    u = u_raw % Q
    return np.where(u >= Q // 2, u - Q, u).astype(np.int64)


def true_linf(vec):
    v = vec[:M].astype(np.int64)
    u = centered_u(vec[M:].astype(np.int64))
    if np.all(v == 0) and np.all(u == 0):
        return 999, 0, 0
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
    """Convert numpy basis to fpylll GSO matrix."""
    m = IntegerMatrix(DIM, DIM)
    for i in range(DIM):
        for j in range(DIM):
            m[i, j] = int(B[i, j])
    mg = GSO.Mat(m, U=IntegerMatrix.identity(DIM),
                 UinvT=IntegerMatrix.identity(DIM))
    mg.update_gso()
    return mg


def load_basis(path):
    B = np.load(path).astype(np.int64)
    ri, li = best_row_info(B)
    print(f"Loaded {os.path.basename(path)}: best linf={li} (row {ri})")
    return B


def extract_basis(g):
    """Extract basis from Siever to numpy."""
    Bn = np.zeros((DIM, DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM):
            Bn[i, j] = int(g.M.B[i, j])
    return Bn


def focused_pump_chain(B_init, chain_id=0, threads=32,
                       beta_sequence=None, max_top_kappa=5):
    """Run pump only on top blocks (kappa=0..max_top_kappa) at increasing β.

    The best row (linf=18) is at index 4. To improve it, we only need to
    process blocks where kappa <= 4. Each block gets a single pump call at
    each β level.
    """
    if beta_sequence is None:
        beta_sequence = [90, 95, 100, 105, 110, 115, 120, 125, 130]

    B_curr = B_init.copy()
    _, best_linf = best_row_info(B_curr)
    best_vec = None
    best_beta = 0

    print(f"\n{'='*60}")
    print(f"Focused Pump Chain {chain_id}: β={beta_sequence}")
    print(f"Top blocks: kappa=0..{max_top_kappa}")
    print(f"Threads: {threads}")
    print(f"Initial linf: {best_linf}")
    print(f"{'='*60}\n")

    t_start = time.time()
    total_pump_calls = 0

    for beta in beta_sequence:
        t1 = time.time()
        dim4free = int(11.5 + 0.075 * beta)
        sieve_dim = beta - dim4free
        print(f"\n--- β={beta} (d4f={dim4free}, sieve_dim={sieve_dim}) ---")

        improved_this_beta = False

        for kappa in range(0, max_top_kappa + 1):
            if kappa + beta > DIM:
                break

            # Determine actual blocksize and dim4free for boundary cases
            actual_beta = min(beta, DIM - kappa)
            lost_dim = beta - actual_beta
            actual_d4f = max(dim4free - lost_dim, 0)

            print(f"  kappa={kappa} β={actual_beta} d4f={actual_d4f} ...", end=' ', flush=True)

            seed_val = chain_id * 100000 + beta * 1000 + kappa
            g = None

            for retry in range(3):
                try:
                    p = SieverParams()
                    p._set("threads", threads)
                    p._set("saturation_ratio", 0.3)
                    p._set("db_size_factor", 3.0)

                    g = Siever(mk_gso_np(B_curr), p, seed=seed_val + retry)

                    pump(g, dummy_tracer, kappa, actual_beta, actual_d4f,
                         down_sieve=False)

                    # LLL after pump (wrapped - our patched version)
                    try:
                        g.lll(0, DIM)
                    except Exception:
                        pass

                    break
                except Exception as e:
                    if retry == 2:
                        print(f"FAIL: {e}", flush=True)
                        g = None
                    else:
                        print(f"(retry {retry})", end=' ', flush=True)

            total_pump_calls += 1

            if g is not None:
                B_curr = extract_basis(g)
                ri, li = best_row_info(B_curr)
                if li < best_linf:
                    best_linf = li
                    best_vec = B_curr[ri].copy()
                    best_beta = beta
                    improved_this_beta = True
                    print(f"linf={li} *** NEW BEST ***", flush=True)

                    # Save immediately on improvement
                    np.save(os.path.join(RESDIR, f"focused_pump_c{chain_id}_best.npy"),
                            B_curr)
                else:
                    print(f"linf={li}", flush=True)

                if li <= GAMMA:
                    print(f"\n!!! SOLVED !!! linf={li} at β={beta}, kappa={kappa}")
                    v = best_vec[:M]
                    u = centered_u(best_vec[M:].astype(np.int64))
                    np.save(os.path.join(RESDIR, f"p3_focused_pump_solved_c{chain_id}.npy"),
                            np.concatenate([v, u]))
                    return best_linf, best_vec
            else:
                print("(skipped)", flush=True)

        dt = time.time() - t1
        print(f"  β={beta} done: {dt:.0f}s, best={best_linf}", flush=True)

        # Save after each beta
        np.save(os.path.join(RESDIR, f"focused_pump_c{chain_id}_b{beta}.npy"), B_curr)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Chain {chain_id} DONE: {elapsed:.0f}s, {total_pump_calls} pump calls")
    print(f"Best: linf={best_linf} at β={best_beta}")
    print(f"{'='*60}")

    return best_linf, best_vec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-id", type=int, default=0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--basis", type=str, default=None)
    ap.add_argument("--beta-start", type=int, default=90)
    ap.add_argument("--beta-end", type=int, default=130)
    ap.add_argument("--max-kappa", type=int, default=5)
    args = ap.parse_args()

    if args.basis:
        bpath = args.basis
    else:
        bpath = os.path.join(RESDIR, "g6k_push_b95_p3.npy")

    B_init = load_basis(bpath)

    betas = list(range(args.beta_start, args.beta_end + 1, 5))
    focused_pump_chain(
        B_init, chain_id=args.chain_id, threads=args.threads,
        beta_sequence=betas, max_top_kappa=args.max_kappa
    )
