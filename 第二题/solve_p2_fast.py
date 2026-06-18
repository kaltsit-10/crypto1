#!/usr/bin/env python3
"""Problem 2 快速求解器：渐进 BKZ + Babai CVP
使用正确格基构造 [I, -A^T; 0, q*I]，每阶段 BKZ 后用 Babai 找 CVP 解。"""

import numpy as np, os, time, json
from datetime import datetime
from fpylll import IntegerMatrix, GSO, LLL, BKZ

BASE = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
Q, M, N, DIM = 100, 100, 100, 200
GAMMA = 15
OUT = os.path.join(BASE, 'results_p2')
os.makedirs(OUT, exist_ok=True)

STRATEGIES = os.path.join(BASE, 'default.json')
BLOCK_SIZES = [40, 50, 55, 60, 65, 70, 75, 80]

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
        f.write(f"# ||v||^2+||u||^2 = {np.linalg.norm(v.astype(float))**2+np.linalg.norm(u.astype(float))**2:.0f}\n")
        f.write(f"# method: {method}\n")
    print(f"*** SOLUTION [{method}] -> {path}")
    return path

def babai_cvp(B_mat):
    """Babai nearest plane CVP on lattice B_mat, target = [0, -t]."""
    gso = GSO.Mat(B_mat)
    gso.update_gso()
    
    b_arr = np.array([[int(B_mat[i,j]) for j in range(DIM)] for i in range(DIM)], dtype=np.float64)
    r = np.array([gso.get_r(i,i) for i in range(DIM)])
    
    # Compute bstar
    bstar = np.zeros((DIM, DIM))
    for i in range(DIM):
        bstar[i] = b_arr[i].copy()
        for j in range(i):
            bstar[i] -= gso.get_mu(i, j) * bstar[j]
    
    # Target: [0, -t] in the lattice
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
    
    # lattice_pt = (v', u') in L. We need (v, u-t) = lattice_pt
    v = lattice_pt[:M]
    u_raw = (lattice_pt[M:DIM] + t) % Q
    u = centered(u_raw)
    
    return v, u

def scan_basis(B_mat):
    """Scan basis directly for short vectors (like Problem 1)."""
    best = 999
    best_v, best_u = None, None
    for i in range(DIM):
        v = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
        u_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
        if np.all(v == 0):
            continue
        u = centered(u_raw)
        lv, lu = int(np.max(np.abs(v))), int(np.max(np.abs(u)))
        s = max(lv, lu)
        if s < best:
            best = s
            best_v, best_u = v.copy(), u.copy()
        if lv <= GAMMA and lu <= GAMMA:
            # Check if this solves A*v + u ≡ t
            check = (A @ v + u_raw - t) % Q
            if np.all(check == 0):
                save_sol(v, u, "basis_scan")
                return v, u, True
    return best_v, best_u, False

print("=" * 60)
print("Problem 2 Fast Solver: Progressive BKZ + Babai CVP")
print(f"q={Q}, dim={DIM}, gamma={GAMMA}")
print("=" * 60)

# Build basis: [I, -A^T; 0, q*I]
print("\nBuilding SIS basis...", flush=True)
B_mat = IntegerMatrix(DIM, DIM)
for i in range(M):
    B_mat[i, i] = 1
    for j in range(N):
        B_mat[i, M+j] = -int(A[j, i])
for j in range(N):
    B_mat[M+j, M+j] = Q

# LLL
print("LLL...", flush=True)
t0 = time.time()
LLL.reduction(B_mat, delta=0.99)
print(f"  done {time.time()-t0:.0f}s", flush=True)

# Babai CVP
v, u = babai_cvp(B_mat)
lv, lu = int(np.max(np.abs(v))), int(np.max(np.abs(u)))
print(f"  Babai: ||v||_inf={lv}, ||u||_inf={lu}, score={max(lv,lu)}", flush=True)

if max(lv,lu) < 30:
    save_sol(v, u, f"LLL-Babai")

# Progressive BKZ
results = []
for bs in BLOCK_SIZES:
    print(f"\nBKZ-{bs}...", flush=True)
    t0 = time.time()
    try:
        BKZ.reduction(B_mat, BKZ.Param(
            block_size=bs,
            strategies=STRATEGIES,
            max_loops=2 if bs >= 55 else 1,
            flags=BKZ.DEFAULT | BKZ.GH_BND if bs >= 55 else BKZ.DEFAULT
        ))
        dt = time.time() - t0
        print(f"  done {dt:.0f}s", flush=True)
        
        # Save basis
        B_np = np.zeros((DIM, DIM), dtype=np.int64)
        for i in range(DIM):
            for j in range(DIM):
                B_np[i,j] = int(B_mat[i,j])
        np.save(os.path.join(BASE, f'bkz{bs}_p2_basis.npy'), B_np)
        
        # Babai CVP
        v, u = babai_cvp(B_mat)
        lv, lu = int(np.max(np.abs(v))), int(np.max(np.abs(u)))
        s = max(lv, lu)
        print(f"  Babai: ||v||_inf={lv}, ||u||_inf={lu}, score={s}", flush=True)
        
        results.append({'bs': bs, 'babai_score': s, 'lv': lv, 'lu': lu, 'time': dt})
        
        if s < 30:
            save_sol(v, u, f"BKZ{bs}-Babai")
        
        if s <= GAMMA:
            print(f"\n*** SOLVED at BKZ-{bs}! ***")
            break
            
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        break

# Save results
with open(os.path.join(OUT, 'babai_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDONE. Results in {OUT}/")
