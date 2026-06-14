import sys, os, time, numpy as np
Q, M, DIM = 120, 120, 240
BASE_DIR = '/home/linux/PycharmProjects/pythonProject/crypto1'
G6K_PATH = "/tmp/G6K-GPU-Tensor-src"
FPLLL_LIB = "/tmp/g6k/g6k-env/lib"
sys.path.insert(0, G6K_PATH)
os.environ.setdefault('LD_LIBRARY_PATH', f"{FPLLL_LIB}:{G6K_PATH}/kernel:{os.environ.get('LD_LIBRARY_PATH', '')}")
from fpylll import IntegerMatrix, GSO, LLL
from g6k import Siever, SieverParams
from g6k.algorithms.pump import pump
from g6k.utils.stats import dummy_tracer

def best_linf(Bn):
    best = 999
    for i in range(DIM):
        v = Bn[i,:M]; u = Bn[i,M:] % Q
        u = np.where(u >= 60, u - Q, u).astype(np.int64)
        best = min(best, max(int(np.max(np.abs(v))), int(np.max(np.abs(u)))))
    return best

def extract(s): 
    B = np.zeros((DIM,DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM): B[i,j] = int(s.M.B[i,j])
    return B

B = np.load(os.path.join(BASE_DIR, 'p4_gpu_best.npy')).astype(np.int64)
print(f"Start: linf={best_linf(B)}")

for kappa, beta, f_val, label in [(0,135,13,"b135"),(0,140,14,"b140"),(5,140,14,"k5-b140")]:
    bl = best_linf(B)
    if bl <= 16:
        print(f"SOLVED! linf={bl}")
        break
    print(f"Pump {label}: k={kappa} b={beta} f={f_val} (current={bl})", flush=True)
    t0 = time.time()
    m = IntegerMatrix(DIM, DIM)
    for i in range(DIM):
        for j in range(DIM): m[i,j] = int(B[i,j])
    mg = GSO.Mat(m, U=IntegerMatrix.identity(DIM), UinvT=IntegerMatrix.identity(DIM))
    mg.update_gso()
    params = SieverParams()
    params._set(b"threads", 4)
    params._set(b"saturation_ratio", 0.3)
    params._set(b"lift_radius", 1.8)
    params._set(b"goal_r0", -1.0)
    params._set(b"db_size_factor", 2.0)
    params._set(b"default_sieve", b"gpu")
    params._set(b"gauss_crossover", 0)
    params._set(b"otf_lift", False)
    g = Siever(mg, params, seed=hash(label)&0x7FFFFFFF)
    try:
        pump(g, dummy_tracer, kappa, beta, f_val, down_sieve=False, goal_r0=None, verbose=False)
        B = extract(g); dt = time.time()-t0
        new_bl = best_linf(B)
        np.save(os.path.join(BASE_DIR, 'p4_gpu_best.npy'), B)
        print(f"  Done: linf={new_bl} ({dt:.0f}s)", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print(f"Final: linf={best_linf(B)}")
