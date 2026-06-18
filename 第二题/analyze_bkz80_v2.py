import numpy as np, os
from fpylll import IntegerMatrix, GSO
from itertools import product

BASE = '/home/dys1013/crypto_challenge'
A = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
Q, M, N, DIM = 100, 100, 100, 200
GAMMA = 15

def centered(u):
    return np.where((u%Q) >= Q//2, (u%Q) - Q, (u%Q)).astype(np.int64)

def score_v(v):
    u_raw = (t - A @ v) % Q
    u = centered(u_raw)
    return max(int(np.max(np.abs(v))), int(np.max(np.abs(u)))), u

B_np = np.load(os.path.join(BASE, 'bkz80_p2_basis_v2.npy'))
B_mat = IntegerMatrix(DIM, DIM)
for i in range(DIM):
    for j in range(DIM):
        B_mat[i,j] = int(B_np[i,j])

# Find top short homogeneous rows (as numpy arrays for speed)
print("Finding short homogeneous rows...")
short_rows = []
for i in range(DIM):
    vb = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
    if np.all(vb == 0): continue
    ub_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
    ub = centered(ub_raw)
    s = max(int(np.max(np.abs(vb))), int(np.max(np.abs(ub))))
    if s <= 25:
        short_rows.append((s, i, vb, ub_raw))
short_rows.sort()
print(f"Found {len(short_rows)} rows with score<=25")
top = short_rows[:20]
print(f"Top 5 scores: {[r[0] for r in top[:5]]}")

# Babai baseline
gso = GSO.Mat(B_mat); gso.update_gso()
b_arr = np.array([[int(B_mat[i,j]) for j in range(DIM)] for i in range(DIM)], dtype=np.float64)
r_arr = np.array([gso.get_r(i,i) for i in range(DIM)])
bstar = np.zeros((DIM,DIM))
for i in range(DIM):
    bstar[i] = b_arr[i].copy()
    for j in range(i):
        bstar[i] -= gso.get_mu(i,j) * bstar[j]

target = np.zeros(DIM, dtype=np.float64)
target[M:DIM] = (-t % Q).astype(np.float64)
t_curr = target.copy()
coeffs = np.zeros(DIM, dtype=np.int64)
for j in range(DIM-1, -1, -1):
    c = round(np.dot(t_curr, bstar[j]) / r_arr[j])
    coeffs[j] = int(c)
    t_curr -= c * b_arr[j]

v0 = np.zeros(M, dtype=np.int64)
for j in range(DIM):
    v0 += int(coeffs[j]) * np.array([int(B_mat[j,i]) for i in range(M)], dtype=np.int64)

s0, u0 = score_v(v0)
print(f"\nBabai baseline: score={s0}")

# Strategy: add small multiples of short homogeneous rows to Babai result
# This is much faster than perturbing Babai coefficients
print(f"\nCombining Babai + {len(top)} short rows (coeffs in [-2,2])...")
best_s, best_v, best_u = s0, v0.copy(), u0
count = 0

# Add single short rows
for s_rank, idx, vb, ub_raw in top:
    for c in range(-2, 3):
        if c == 0: continue
        v_try = v0 + c * vb
        s, u = score_v(v_try)
        if s < best_s:
            best_s = s
            best_v, best_u = v_try.copy(), u.copy()
            print(f"  + {c}*row{idx} (score={s_rank}): score={s}", flush=True)
            if s <= GAMMA:
                break
        count += 1
    if best_s <= GAMMA:
        break

# Add pairs of short rows
if best_s > GAMMA:
    print(f"\nPairs of short rows (coeffs in [-1,1])...")
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            s_i, idx_i, vb_i, ub_i = top[i]
            s_j, idx_j, vb_j, ub_j = top[j]
            for ci, cj in product([-1, 0, 1], repeat=2):
                if ci == 0 and cj == 0: continue
                v_try = v0 + ci * vb_i + cj * vb_j
                s, u = score_v(v_try)
                if s < best_s:
                    best_s = s
                    best_v, best_u = v_try.copy(), u.copy()
                    print(f"  +{ci}*row{idx_i} +{cj}*row{idx_j}: score={s}", flush=True)
                count += 1
                if best_s <= GAMMA:
                    break
            if best_s <= GAMMA:
                break
        if best_s <= GAMMA:
            break

print(f"\nFinal best: score={best_s} (combos checked: {count})")

if best_s <= GAMMA:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BASE, 'results_p2', f'solution_p2_{ts}.txt')
    with open(out, 'w') as f:
        f.write(f"v = {best_v.tolist()}\nu = {best_u.tolist()}\n")
        f.write(f"# ||v||_inf = {int(np.max(np.abs(best_v)))}\n")
        f.write(f"# ||u||_inf = {int(np.max(np.abs(best_u)))}\n")
        f.write(f"# method: BKZ80-Babai+shortrows\n")
    print(f"*** SOLUTION! {out} ***")
