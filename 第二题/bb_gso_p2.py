#!/usr/bin/env python3
"""
B&B CVP with GSO pruning — depth=8, range=±4.
Iterates over last 8 coefficients, uses GSO l2 bound to prune.
"""
import numpy as np
from fpylll import IntegerMatrix, GSO
from itertools import product
import os, time, math
from datetime import datetime

Q, MD, ND, DIM = 100, 100, 100, 200
GAMMA = 15
BASE = '/home/dys1013/crypto_challenge'
A_mat = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t_vec = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
OUT = os.path.join(BASE, 'results_p2')
os.makedirs(OUT, exist_ok=True)

def centered(u):
    return np.where((u % Q) >= Q // 2, (u % Q) - Q, (u % Q)).astype(np.int64)

B_np = np.load(os.path.join(BASE, 'bkz80_p2_basis_v2.npy'))
B_mat = IntegerMatrix(DIM, DIM)
for i in range(DIM):
    for j in range(DIM):
        B_mat[i, j] = int(B_np[i, j])

gso = GSO.Mat(B_mat); gso.update_gso()
b_arr = np.array([[float(B_mat[i,j]) for j in range(DIM)] for i in range(DIM)])
r_diag = np.array([gso.get_r(i,i) for i in range(DIM)])
bstar = np.zeros((DIM, DIM))
for i in range(DIM):
    bstar[i] = b_arr[i].copy()
    for j in range(i):
        bstar[i] -= gso.get_mu(i,j) * bstar[j]

# Babai
target = np.zeros(DIM, dtype=np.float64)
target[MD:DIM] = (-t_vec % Q).astype(np.float64)
t_curr = target.copy()
coeffs = np.zeros(DIM, dtype=np.int64)
for j in range(DIM-1, -1, -1):
    c = round(np.dot(t_curr, bstar[j]) / r_diag[j])
    coeffs[j] = c
    t_curr -= c * b_arr[j]

row_v = np.zeros((DIM, MD), dtype=np.int64)
A_row = np.zeros((DIM, ND), dtype=np.int64)
for i in range(DIM):
    row_v[i] = np.array([int(B_mat[i,j]) for j in range(MD)], dtype=np.int64)
    A_row[i] = A_mat @ row_v[i]

# Baseline
v0 = np.zeros(MD, dtype=np.int64)
Av0 = np.zeros(ND, dtype=np.int64)
for j in range(DIM):
    v0 += int(coeffs[j]) * row_v[j]
    Av0 += int(coeffs[j]) * A_row[j]
s0 = max(int(np.max(np.abs(v0))), int(np.max(np.abs(centered((t_vec-Av0)%Q)))))
print(f"Babai: score={s0}", flush=True)

best_score = s0
best_l2_sq = float(s0 * s0) * DIM  # l2^2 bound

# SEARCH_DEPTH last coefficients, range ±SEARCH_RANGE
SEARCH_DEPTH = 8
SEARCH_RANGE = 4
START_IDX = DIM - SEARCH_DEPTH

total = (2*SEARCH_RANGE+1)**SEARCH_DEPTH
print(f"Search: last {SEARCH_DEPTH} coeffs ±{SEARCH_RANGE} = {total:,} combos", flush=True)

def quick_lb2(t_eff):
    """Lower bound on l2^2 from dims 0..START_IDX-1."""
    lb = 0.0
    for j in range(START_IDX):
        proj = np.dot(t_eff, bstar[j]) / r_diag[j]
        frac = abs(proj - round(proj))
        lb += r_diag[j] * frac * frac
    return lb

nodes = pruned = improvements = 0
t0 = time.time()

for combo in product(range(-SEARCH_RANGE, SEARCH_RANGE+1), repeat=SEARCH_DEPTH):
    nodes += 1
    
    t_eff = target.copy()
    v_top = np.zeros(MD, dtype=np.int64)
    Av_top = np.zeros(ND, dtype=np.int64)
    
    for k, delta in enumerate(combo):
        j = START_IDX + k
        c = int(coeffs[j]) + delta
        t_eff -= c * b_arr[j]
        v_top += c * row_v[j]
        Av_top += c * A_row[j]
    
    # Prune by GSO l2 lower bound
    lb2 = quick_lb2(t_eff)
    if lb2 > best_l2_sq:
        pruned += 1
        continue
    
    # Full evaluation: Babai for lower dims
    v_low = np.zeros(MD, dtype=np.int64)
    Av_low = np.zeros(ND, dtype=np.int64)
    t = t_eff.copy()
    for j in range(START_IDX-1, -1, -1):
        c = round(np.dot(t, bstar[j]) / r_diag[j])
        t -= c * b_arr[j]
        v_low += c * row_v[j]
        Av_low += c * A_row[j]
    
    v_full = v_top + v_low
    Av_full = Av_top + Av_low
    u_full = centered((t_vec - Av_full) % Q)
    lv = int(np.max(np.abs(v_full)))
    lu = int(np.max(np.abs(u_full)))
    s = max(lv, lu)
    
    if s < best_score:
        best_score = s
        best_l2_sq = float(s * s) * DIM
        improvements += 1
        print(f"  score={s} (lv={lv}, lu={lu})", flush=True)
        if s <= GAMMA:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(OUT, f'solution_p2_{ts}.txt')
            with open(path, 'w') as f:
                f.write(f"v = {v_full.tolist()}\nu = {u_full.tolist()}\n")
                f.write(f"# ||v||_inf = {lv}\n# ||u||_inf = {lu}\n")
                f.write(f"# method: bb-enum\n# constraint_verified: True\n")
            print(f"*** SOLVED! → {path} ***", flush=True)
            import sys; sys.exit(0)
    
    if nodes % 5000000 == 0:
        dt = time.time() - t0
        pct = 100.0 * nodes / total
        print(f"  {nodes/1e6:.1f}M/{total/1e6:.0f}M ({pct:.0f}%) best={best_score} pruned={pruned} {nodes/dt:.0f}/s", flush=True)

dt = time.time() - t0
print(f"\nDone: {dt:.0f}s, {nodes} nodes, {pruned} pruned, {improvements} improvements", flush=True)
print(f"Best: score={best_score}", flush=True)
