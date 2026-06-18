#!/usr/bin/env python3
"""Continue BKZ from 75 to 80 on Problem 2, then Babai CVP."""
import numpy as np, os, time, json
from datetime import datetime
from fpylll import IntegerMatrix, GSO, BKZ

BASE = '/home/dys1013/crypto_challenge'
A = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
Q, M, N, DIM = 100, 100, 100, 200
GAMMA = 15
STRATEGIES = os.path.join(BASE, 'default.json')
OUT = os.path.join(BASE, 'results_p2')
os.makedirs(OUT, exist_ok=True)

def centered(u):
    u = u % Q
    return np.where(u >= Q//2, u - Q, u).astype(np.int64)

def save_sol(v, u, method):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT, f"solution_p2_{ts}.txt")
    with open(path, 'w') as f:
        f.write(f"v = {v.tolist()}\nu = {u.tolist()}\n")
        f.write(f"# ||v||_inf = {int(np.max(np.abs(v)))}\n")
        f.write(f"# ||u||_inf = {int(np.max(np.abs(u)))}\n")
        f.write(f"# method: {method}\n")
    print(f"*** SOLUTION [{method}] -> {path}", flush=True)
    return path

def babai_cvp(B_mat):
    gso = GSO.Mat(B_mat)
    gso.update_gso()
    b_arr = np.array([[int(B_mat[i,j]) for j in range(DIM)] for i in range(DIM)], dtype=np.float64)
    r = np.array([gso.get_r(i,i) for i in range(DIM)])
    bstar = np.zeros((DIM, DIM))
    for i in range(DIM):
        bstar[i] = b_arr[i].copy()
        for j in range(i):
            bstar[i] -= gso.get_mu(i, j) * bstar[j]
    target = np.zeros(DIM, dtype=np.float64)
    target[M:DIM] = (-t % Q).astype(np.float64)
    t_curr = target.copy()
    coeffs = np.zeros(DIM)
    for j in range(DIM-1, -1, -1):
        c = round(np.dot(t_curr, bstar[j]) / r[j])
        coeffs[j] = c
        t_curr -= c * b_arr[j]
    lattice_pt = np.zeros(DIM, dtype=np.int64)
    for j in range(DIM):
        lattice_pt += int(coeffs[j]) * np.array([int(B_mat[j,i]) for i in range(DIM)], dtype=np.int64)
    v = lattice_pt[:M]
    u_raw = (lattice_pt[M:DIM] + t) % Q
    u = centered(u_raw)
    return v, u

# Load BKZ-75 basis
print("Loading BKZ-75 basis...", flush=True)
B_np = np.load(os.path.join(BASE, 'bkz75_p2_basis.npy'))
B_mat = IntegerMatrix(DIM, DIM)
for i in range(DIM):
    for j in range(DIM):
        B_mat[i,j] = int(B_np[i,j])

# Quick scan
print("Quick scan of BKZ-75 basis...", flush=True)
best_s, best_v, best_u = 999, None, None
for i in range(DIM):
    vb = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
    if np.all(vb == 0): continue
    ub_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
    ub = centered(ub_raw)
    s = max(int(np.max(np.abs(vb))), int(np.max(np.abs(ub))))
    if s < best_s:
        best_s = s
    # Check if homogeneous vector happens to solve inhomogeneous
    check = (A @ vb + ub_raw - t) % Q
    if np.all(check == 0) and max(int(np.max(np.abs(vb))), int(np.max(np.abs(ub)))) <= GAMMA:
        save_sol(vb, ub, "homogeneous_by_luck")
        print("FOUND SOLUTION BY LUCK!", flush=True)
print(f"Best homogeneous row: score={best_s}", flush=True)

# Babai on BKZ-75
v, u = babai_cvp(B_mat)
lv, lu = int(np.max(np.abs(v))), int(np.max(np.abs(u)))
s = max(lv, lu)
print(f"BKZ-75 Babai: score={s} (lv={lv}, lu={lu})", flush=True)
if s <= GAMMA:
    save_sol(v, u, "BKZ75-Babai")

# BKZ-80
print("\n=== BKZ-80 ===", flush=True)
t0 = time.time()
try:
    BKZ.reduction(B_mat, BKZ.Param(
        block_size=80,
        strategies=STRATEGIES,
        max_loops=2,
        flags=BKZ.DEFAULT | BKZ.GH_BND
    ))
    dt = time.time() - t0
    print(f"BKZ-80 done: {dt:.0f}s", flush=True)
    
    # Save basis
    B_out = np.zeros((DIM, DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM):
            B_out[i,j] = int(B_mat[i,j])
    np.save(os.path.join(BASE, 'bkz80_p2_basis.npy'), B_out)
    print("BKZ-80 basis saved.", flush=True)
    
    # Babai
    v, u = babai_cvp(B_mat)
    lv, lu = int(np.max(np.abs(v))), int(np.max(np.abs(u)))
    s = max(lv, lu)
    print(f"BKZ-80 Babai: score={s} (lv={lv}, lu={lu})", flush=True)
    
    if s <= GAMMA:
        save_sol(v, u, "BKZ80-Babai")
    elif s < 30:
        save_sol(v, u, f"BKZ80-Babai-s{s}")

    # Scan basis
    print("Scanning BKZ-80 basis...", flush=True)
    for i in range(DIM):
        vb = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
        if np.all(vb == 0): continue
        ub_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
        ub = centered(ub_raw)
        check = (A @ vb + ub_raw - t) % Q
        if np.all(check == 0):
            lv_i, lu_i = int(np.max(np.abs(vb))), int(np.max(np.abs(ub)))
            si = max(lv_i, lu_i)
            print(f"  Row {i}: score={si}", flush=True)
            if si <= GAMMA:
                save_sol(vb, ub, f"BKZ80-scan-row{i}")
    
    print("Done!", flush=True)
except Exception as e:
    print(f"BKZ-80 FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()
