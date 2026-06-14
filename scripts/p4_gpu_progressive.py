#!/usr/bin/env python3
"""P4: GPU-accelerated Progressive BKZ — flatten GS gradually."""
import sys, os, time, numpy as np, ast

Q, M, DIM = 120, 120, 240
GAMMA = 16
BASE_DIR = '/home/linux/PycharmProjects/pythonProject/crypto1'
OUT_DIR = os.path.join(BASE_DIR, 'p4_bases')
os.makedirs(OUT_DIR, exist_ok=True)

G6K_PATH = "/tmp/G6K-GPU-Tensor-src"
FPLLL_LIB = "/tmp/g6k/g6k-env/lib"
sys.path.insert(0, G6K_PATH)
os.environ.setdefault('LD_LIBRARY_PATH',
    f"{FPLLL_LIB}:{G6K_PATH}/kernel:{os.environ.get('LD_LIBRARY_PATH', '')}")

from fpylll import IntegerMatrix, GSO, LLL
from g6k import Siever, SieverParams
from g6k.algorithms.pump import pump
from g6k.utils.stats import dummy_tracer

# Load problem
P4_DIR = os.path.join(BASE_DIR, 'p4_deliver/p4_deliver')
with open(os.path.join(P4_DIR, 'crypto1_repo/第一题/sis_inf_problems/problem4.txt')) as f:
    lines = f.read().strip().split('\n')
A_mat = np.array(ast.literal_eval(lines[0].split('=',1)[1].strip()), dtype=np.int64)
t_vec = np.array(ast.literal_eval(lines[1].split('=',1)[1].strip()), dtype=np.int64)

def check_linf(vv):
    v = vv[:M].astype(np.int64)
    ur = vv[M:].astype(np.int64)
    u = np.where((ur % Q) >= Q//2, (ur % Q)-Q, ur % Q).astype(np.int64)
    if np.all(v==0) and np.all(u==0): return 999
    return max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))

def best_linf(Bn):
    return min(check_linf(np.array([int(Bn[i,j]) for j in range(DIM)], dtype=np.int64))
               for i in range(DIM))

def analyze_gs(B, label=""):
    Bf = B.astype(np.float64)
    gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM))
    for i in range(DIM):
        gso_b[i] = Bf[i].copy()
        for j in range(i):
            if gso_r[j] > 0:
                mu = np.dot(Bf[i], gso_b[j]) / gso_r[j]
                gso_b[i] -= mu * gso_b[j]
        gso_r[i] = np.dot(gso_b[i], gso_b[i])
    hl = best_linf(B)
    target = np.zeros(DIM, dtype=np.float64)
    target[M:] = (-t_vec).astype(np.float64)
    coeffs = np.zeros(DIM, dtype=np.int64); t_rem = target.copy()
    for j in range(DIM-1, -1, -1):
        if gso_r[j] > 0:
            c = int(round(np.dot(t_rem, gso_b[j]) / gso_r[j]))
            coeffs[j] = c; t_rem -= float(c) * Bf[j]
    vec = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        if coeffs[j]!=0: vec += float(coeffs[j]) * Bf[j]
    v = np.round(vec[:M]).astype(np.int64); u_lat = np.round(vec[M:]).astype(np.int64)
    u_p4 = (t_vec + u_lat) % Q; u_p4 = np.where(u_p4>=60, u_p4-Q, u_p4).astype(np.int64)
    babai = max(int(np.max(np.abs(v))), int(np.max(np.abs(u_p4))))
    d50_100 = gso_r[100]/gso_r[50] if gso_r[50]>0 else 0
    print(f"  [{label:10s}] homo={hl:2d} Babai={babai:2d} GS[0]={gso_r[0]:.0f} GS[50]={gso_r[50]:.0f} GS[100]={gso_r[100]:.0f} GS[150]={gso_r[150]:.1f} d50→100={d50_100:.3f}")
    return hl, babai

def extract_basis(siever):
    B = np.zeros((DIM,DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM): B[i,j] = int(siever.M.B[i,j])
    return B

def make_params():
    p = SieverParams()
    p._set(b"threads", 4)
    p._set(b"saturation_ratio", 0.3)
    p._set(b"lift_radius", 1.8)
    p._set(b"goal_r0", -1.0)
    p._set(b"db_size_factor", 3.0)
    p._set(b"default_sieve", b"gpu")
    p._set(b"gauss_crossover", 0)
    p._set(b"otf_lift", False)
    return p

def gpu_pump(B_in, kappa, beta, tours, label):
    """GPU-accelerated BKZ tours."""
    B = B_in.copy()
    best_hl = best_linf(B)
    for t in range(tours):
        t0 = time.time()
        m = IntegerMatrix(DIM, DIM)
        for i in range(DIM):
            for j in range(DIM): m[i,j] = int(B[i,j])
        mg = GSO.Mat(m, U=IntegerMatrix.identity(DIM), UinvT=IntegerMatrix.identity(DIM))
        mg.update_gso()

        g = Siever(mg, make_params(), seed=hash(f"{label}t{t}") & 0x7FFFFFFF)
        f_val = max(0, int(beta*0.12)-2)
        try:
            pump(g, dummy_tracer, kappa, beta, f_val,
                 down_sieve=False, goal_r0=None, verbose=False)
            B = extract_basis(g)
            hl = best_linf(B)
            dt = time.time()-t0
            status = "***" if hl < best_hl else ""
            if hl < best_hl: best_hl = hl
            print(f"    Tour {t}: linf={hl} {status} ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"    Tour {t}: ERROR {e}", flush=True)
            try: B = extract_basis(g)
            except: pass
    return B

# ================================================================
print("="*60)
print("P4 GPU Progressive BKZ — flatten GS cliff")
print("="*60)

# Load linf=17 basis
B = np.load(os.path.join(BASE_DIR, 'p4_gpu_best.npy')).astype(np.int64)
print(f"\nInitial: linf={best_linf(B)}")
analyze_gs(B, "initial")

# Progressive BKZ schedule
beta_schedule = [
    (0, 33, 3, "b33"),
    (0, 40, 3, "b40"),
    (0, 50, 3, "b50"),
    (0, 60, 2, "b60"),
    (0, 70, 2, "b70"),
    (0, 80, 2, "b80"),
    (0, 90, 2, "b90"),
    (0, 100, 2, "b100"),
    (0, 105, 2, "b105"),
    (0, 110, 2, "b110"),
]

print(f"\nProgressive BKZ:")
total_t0 = time.time()
for kappa, beta, tours, label in beta_schedule:
    print(f"\n  β={beta} ({tours} tours)...", flush=True)
    B = gpu_pump(B, kappa, beta, tours, label)
    analyze_gs(B, label)
    np.save(os.path.join(OUT_DIR, f"p4_{label}.npy"), B)
    np.save(os.path.join(BASE_DIR, 'p4_gpu_best.npy'), B)

    if best_linf(B) <= GAMMA:
        print(f"\n*** Homogeneous basis reached linf<={GAMMA}! ***")
        break

dt = time.time() - total_t0
print(f"\nGPU Progressive BKZ done in {dt:.0f}s")
print(f"Bases saved to {OUT_DIR}/")
