#!/usr/bin/env python3
"""P4: Numba-accelerated Klein CVP worker using delta=0.9999 + homo=16 hybrid basis."""
import numpy as np, ast, os, time, sys
from numba import njit

Q, M, DIM = 120, 120, 240
GAMMA = 16

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
save_path = sys.argv[2] if len(sys.argv) > 2 else 'p4_saves/kh16_numba.npy'
label = os.path.basename(save_path).replace('.npy','')

BASE_DIR = '/home/linux/PycharmProjects/pythonProject/crypto1'
P4_DIR = os.path.join(BASE_DIR, 'p4_deliver/p4_deliver')

with open(os.path.join(P4_DIR, 'crypto1_repo/第一题/sis_inf_problems/problem4.txt')) as f:
    lines = f.read().strip().split('\n')
A_mat = np.array(ast.literal_eval(lines[0].split('=',1)[1].strip()), dtype=np.int64)
t_vec = np.array(ast.literal_eval(lines[1].split('=',1)[1].strip()), dtype=np.int64)

# Load hybrid basis: BKZ-100 row0 replaced by homo=16 vector, then deep LLL delta=0.9999
B = np.load(os.path.join(BASE_DIR, 'p4_saves/basis_d09999_h16.npy')).astype(np.int64)
Bf = B.astype(np.float64)

# Check homo linf
best_hl = 999
for i in range(DIM):
    v = B[i,:M]; u = B[i,M:] % Q; u = np.where(u>=60, u-Q, u).astype(np.int64)
    best_hl = min(best_hl, max(int(np.max(np.abs(v))), int(np.max(np.abs(u)))))
print(f"[{label}] Hybrid basis homo linf={best_hl}")

# GSO
gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM)); mu = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_b[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0:
            mu[i,j] = np.dot(Bf[i], gso_b[j]) / gso_r[j]
            gso_b[i] -= mu[i,j] * gso_b[j]
    gso_r[i] = np.dot(gso_b[i], gso_b[i])

print(f"[{label}] GS[0]={gso_r[0]:.0f} GS[50]={gso_r[50]:.0f} GS[100]={gso_r[100]:.0f} d50->100={np.sqrt(gso_r[100]/gso_r[50]):.4f}")

target = np.zeros(DIM, dtype=np.float64); target[M:] = (-t_vec).astype(np.float64)

# Precompute data for numba
A_nb = A_mat.astype(np.int64)
t_nb = t_vec.astype(np.int64)
B_nb = Bf.copy()
gso_r_nb = gso_r.copy()
gso_b_nb = gso_b.copy()
mu_nb = mu.copy()
target_nb = target.copy()

@njit
def klein_trial(target_arr, B_arr, gso_r_arr, gso_b_arr, rand_vals):
    coeffs = np.zeros(DIM, dtype=np.int64)
    t_rem = target_arr.copy()
    ri = 0
    for j in range(DIM-1, -1, -1):
        r_sq = gso_r_arr[j]
        if r_sq > 0:
            c_exact = 0.0
            for k in range(DIM):
                c_exact += t_rem[k] * gso_b_arr[j, k]
            c_exact /= r_sq
            if j >= 200: c = int(round(c_exact + rand_vals[ri])); ri += 1
            elif j >= 180: c = int(round(c_exact + rand_vals[ri])); ri += 1
            elif j >= 160: c = int(round(c_exact + rand_vals[ri])); ri += 1
            elif j >= 120: c = int(round(c_exact + rand_vals[ri])); ri += 1
            else: c = int(round(c_exact))
            coeffs[j] = c
            for k in range(DIM):
                t_rem[k] -= float(c) * B_arr[j, k]
    return coeffs

@njit
def reconstruct_and_check(coeffs, B_arr, A_arr, t_arr):
    vec = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        c = coeffs[j]
        if c != 0:
            for k in range(DIM):
                vec[k] += float(c) * B_arr[j, k]
    v = np.zeros(M, dtype=np.int64)
    for i in range(M): v[i] = int(round(vec[i]))
    ul = np.zeros(M, dtype=np.int64)
    for i in range(M): ul[i] = int(round(vec[M + i]))
    u = np.zeros(M, dtype=np.int64)
    for i in range(M):
        val = (t_arr[i] + ul[i]) % Q
        u[i] = val - Q if val >= 60 else val
    ok = True
    for i in range(M):
        s = 0
        for j in range(M): s += A_arr[i, j] * v[j]
        if (s + u[i]) % Q != t_arr[i] % Q:
            ok = False; break
    if not ok: return 999
    lv = 0
    for i in range(M):
        av = abs(v[i])
        if av > lv: lv = av
    lu = 0
    for i in range(M):
        au = abs(u[i])
        if au > lu: lu = au
    return max(lv, lu)

print(f"[{label}] Numba Klein seed={seed}")

best = 999
if os.path.exists(save_path):
    try:
        old = np.load(save_path)
        best = max(int(np.max(np.abs(old[:M]))), int(np.max(np.abs(old[M:]))))
        print(f"[{label}] Resumed: best={best}")
    except: pass

rng = np.random.RandomState(seed)
t0 = time.time(); trial = 0
RAND_BUF_SIZE = 100000

while True:
    rand_buffer = np.zeros((RAND_BUF_SIZE, 4), dtype=np.float64)
    for i in range(RAND_BUF_SIZE):
        rand_buffer[i, 0] = rng.normal(0, 1.0)
        rand_buffer[i, 1] = rng.normal(0, 0.7)
        rand_buffer[i, 2] = rng.normal(0, 0.5)
        rand_buffer[i, 3] = rng.normal(0, 0.3)
    for i in range(RAND_BUF_SIZE):
        trial += 1
        coeffs = klein_trial(target_nb, B_nb, gso_r_nb, gso_b_nb, rand_buffer[i])
        li = reconstruct_and_check(coeffs, B_nb, A_nb, t_nb)
        if li < best:
            best = li
            dt = time.time() - t0
            vec = np.zeros(DIM, dtype=np.float64)
            for j in range(DIM):
                if coeffs[j] != 0: vec += float(coeffs[j]) * Bf[j]
            v = np.round(vec[:M]).astype(np.int64)
            ul = np.round(vec[M:]).astype(np.int64)
            u = (t_vec + ul) % Q; u = np.where(u >= 60, u - Q, u).astype(np.int64)
            print(f"[{label}] t{trial} linf={li} |v|={np.max(np.abs(v))} |u|={np.max(np.abs(u))} ({dt:.0f}s)")
            np.save(save_path, np.concatenate([v, u]))
            if best <= GAMMA:
                print(f"*** SOLVED! ***"); sys.exit(0)
        if trial % 500000 == 0:
            dt = time.time() - t0
            print(f"[{label}] {trial} best={best} {trial/dt:.0f}/s")
