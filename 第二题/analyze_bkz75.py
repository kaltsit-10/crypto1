import numpy as np, os, time

BASE = '/home/dys1013/crypto_challenge'
A = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
Q, M, N, DIM = 100, 100, 100, 200

def centered(u):
    return np.where((u%Q) >= Q//2, (u%Q) - Q, (u%Q)).astype(np.int64)

B75 = np.load(os.path.join(BASE, 'bkz75_p2_basis.npy'))
B70 = np.load(os.path.join(BASE, 'bkz70_p2_basis.npy'))

print("=== Homogeneous Row Quality ===")
for label, B in [("BKZ-70", B70), ("BKZ-75", B75)]:
    best_s, best_i, best_lv, best_lu = 999, -1, 0, 0
    scores = []
    for i in range(DIM):
        vb = np.array([int(B[i,j]) for j in range(M)], dtype=np.int64)
        ub_raw = np.array([int(B[i,j]) for j in range(M, DIM)], dtype=np.int64)
        if np.all(vb == 0): continue
        ub = centered(ub_raw)
        lvi, lui = int(np.max(np.abs(vb))), int(np.max(np.abs(ub)))
        si = max(lvi, lui)
        scores.append(si)
        if si < best_s:
            best_s, best_i, best_lv, best_lu = si, i, lvi, lui
    
    scores.sort()
    print(f"\n{label}:")
    print(f"  Best row: idx={best_i}, score={best_s} (lv={best_lv}, lu={best_lu})")
    print(f"  Top 10 scores: {scores[:10]}")
    print(f"  Rows with score<=20: {sum(1 for s in scores if s <= 20)}")
    print(f"  Rows with score<=25: {sum(1 for s in scores if s <= 25)}")

print("\n=== Full Babai + Enumeration on BKZ-75 ===")
from fpylll import IntegerMatrix, GSO

B_mat = IntegerMatrix(DIM, DIM)
for i in range(DIM):
    for j in range(DIM):
        B_mat[i,j] = int(B75[i,j])

gso = GSO.Mat(B_mat); gso.update_gso()
b_arr = np.array([[int(B_mat[i,j]) for j in range(DIM)] for i in range(DIM)], dtype=np.float64)
r = np.array([gso.get_r(i,i) for i in range(DIM)])
bstar = np.zeros((DIM,DIM))
for i in range(DIM):
    bstar[i] = b_arr[i].copy()
    for j in range(i):
        bstar[i] -= gso.get_mu(i,j) * bstar[j]

target = np.zeros(DIM, dtype=np.float64)
target[M:DIM] = (-t % Q).astype(np.float64)
t_curr = target.copy()
coeffs = np.zeros(DIM)
for j in range(DIM-1, -1, -1):
    c = round(np.dot(t_curr, bstar[j]) / r[j])
    coeffs[j] = c
    t_curr -= c * b_arr[j]

v_babai = np.zeros(M, dtype=np.int64)
for j in range(DIM):
    v_babai += int(coeffs[j]) * np.array([int(B_mat[j,i]) for i in range(M)], dtype=np.int64)

def score_v(v):
    u_raw = (t - A @ v) % Q
    u = centered(u_raw)
    return max(int(np.max(np.abs(v))), int(np.max(np.abs(u)))), u

s0, u0 = score_v(v_babai)
print(f"Babai: score={s0}")

# Enumerate +/-3 on last 3 coeffs
best_s, best_v = s0, v_babai.copy()
for d1 in range(-3, 4):
    for d2 in range(-3, 4):
        for d3 in range(-3, 4):
            v_try = v_babai.copy()
            for j in range(DIM):
                delta = 0
                if j == DIM-1: delta = d3
                elif j == DIM-2: delta = d2
                elif j == DIM-3: delta = d1
                if delta != 0:
                    v_try += int(delta) * np.array([int(B_mat[j,i]) for i in range(M)], dtype=np.int64)
            s, u = score_v(v_try)
            if s < best_s:
                best_s = s
                best_v = v_try.copy()
                if s <= 20:
                    print(f"  Improved: score={s}")
print(f"Best after enum: score={best_s}")

# Single row enumeration
print("\n=== Single Row Enum (k in [-7,7]) ===")
best_s2, best_info = 999, None
for i in range(DIM):
    vb = np.array([int(B75[i,j]) for j in range(M)], dtype=np.int64)
    ub_raw = np.array([int(B75[i,j]) for j in range(M, DIM)], dtype=np.int64)
    if np.all(vb == 0): continue
    for k in range(-7, 8):
        if k == 0: continue
        v_sol = k * vb
        u_sol_raw = (k * ub_raw + t) % Q
        u_sol = centered(u_sol_raw)
        lv, lu = int(np.max(np.abs(v_sol))), int(np.max(np.abs(u_sol)))
        s = max(lv, lu)
        if s < best_s2:
            best_s2 = s
            best_info = (i, k, lv, lu)
print(f"Best single-row: row={best_info[0]}, k={best_info[1]}, score={best_s2} (lv={best_info[2]}, lu={best_info[3]})")

# 2-row combination
print("\n=== Top 5 two-row combos (c1,c2 in [-2,2]) ===")
rows = []
for i in range(DIM):
    vb = np.array([int(B75[i,j]) for j in range(M)], dtype=np.int64)
    ub_raw = np.array([int(B75[i,j]) for j in range(M, DIM)], dtype=np.int64)
    if np.all(vb == 0): continue
    ub = centered(ub_raw)
    s = max(int(np.max(np.abs(vb))), int(np.max(np.abs(ub))))
    rows.append((s, i))
rows.sort()
top5 = [r[1] for r in rows[:5]]
print(f"Top 5 rows: scores={[rows[i][0] for i in range(5)]}")

from itertools import product
best_s3, best_info3 = 999, None
for (i1, i2), (c1, c2) in product(enumerate(top5), repeat=2):
    if i1 >= i2: continue
    for c1v, c2v in product(range(-2, 3), repeat=2):
        if c1v == 0 and c2v == 0: continue
        r1, r2 = top5[i1], top5[i2]
        v_sol = c1v * np.array([int(B75[r1,j]) for j in range(M)], dtype=np.int64) + \
                c2v * np.array([int(B75[r2,j]) for j in range(M)], dtype=np.int64)
        u_raw = (c1v * np.array([int(B75[r1,j]) for j in range(M,DIM)], dtype=np.int64) + \
                 c2v * np.array([int(B75[r2,j]) for j in range(M,DIM)], dtype=np.int64) + t) % Q
        u_sol = centered(u_raw)
        lv, lu = int(np.max(np.abs(v_sol))), int(np.max(np.abs(u_sol)))
        s = max(lv, lu)
        if s < best_s3:
            best_s3 = s
            best_info3 = (r1, r2, c1v, c2v, lv, lu)
print(f"Best 2-row: rows=({best_info3[0]},{best_info3[1]}), c=({best_info3[2]},{best_info3[3]}), score={best_s3} (lv={best_info3[4]}, lu={best_info3[5]})")

print("\nDone.")
