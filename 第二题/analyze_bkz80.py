import numpy as np, os
from fpylll import IntegerMatrix, GSO

BASE = '/home/dys1013/crypto_challenge'
A = np.load(os.path.join(BASE, 'problem2_A.npy')).astype(np.int64)
t = np.load(os.path.join(BASE, 'problem2_t.npy')).astype(np.int64)
Q, M, N, DIM = 100, 100, 100, 200
GAMMA = 15

def centered(u):
    return np.where((u%Q) >= Q//2, (u%Q) - Q, (u%Q)).astype(np.int64)

B_np = np.load(os.path.join(BASE, 'bkz80_p2_basis_v2.npy'))
B_mat = IntegerMatrix(DIM, DIM)
for i in range(DIM):
    for j in range(DIM):
        B_mat[i,j] = int(B_np[i,j])

# Homogeneous scan
print("=== Homogeneous Row Scan ===")
best_s, best_i, best_lv, best_lu = 999, -1, 0, 0
scores = []
for i in range(DIM):
    vb = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
    ub_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
    if np.all(vb == 0): continue
    ub = centered(ub_raw)
    lv, lu = int(np.max(np.abs(vb))), int(np.max(np.abs(ub)))
    s = max(lv, lu)
    scores.append(s)
    if s < best_s:
        best_s, best_i, best_lv, best_lu = s, i, lv, lu
scores.sort()
print(f"Best row: idx={best_i}, score={best_s} (lv={best_lv}, lu={best_lu})")
print(f"Top 10: {scores[:10]}")
print(f"Rows score<=15: {sum(1 for s in scores if s<=15)}")
print(f"Rows score<=20: {sum(1 for s in scores if s<=20)}")

# Babai CVP
print("\n=== Babai CVP ===")
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
coeffs = np.zeros(DIM, dtype=np.int64)
for j in range(DIM-1, -1, -1):
    c = round(np.dot(t_curr, bstar[j]) / r[j])
    coeffs[j] = int(c)
    t_curr -= c * b_arr[j]

v_babai = np.zeros(M, dtype=np.int64)
for j in range(DIM):
    v_babai += int(coeffs[j]) * np.array([int(B_mat[j,i]) for i in range(M)], dtype=np.int64)

u_raw = (t - A @ v_babai) % Q
u_babai = centered(u_raw)
lv, lu = int(np.max(np.abs(v_babai))), int(np.max(np.abs(u_babai)))
s = max(lv, lu)
print(f"Babai: score={s} (lv={lv}, lu={lu})")

if s <= GAMMA:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BASE, 'results_p2', f'solution_p2_{ts}.txt')
    with open(out, 'w') as f:
        f.write(f"v = {v_babai.tolist()}\nu = {u_babai.tolist()}\n")
        f.write(f"# ||v||_inf = {lv}\n# ||u||_inf = {lu}\n")
        f.write(f"# method: BKZ80-Babai\n")
    print(f"*** SOLUTION! {out} ***")

# Inhomogeneous scan (check each row for CVP solution)
print("\n=== Inhomogeneous Row Scan ===")
best_s2, best_info = 999, None
for i in range(DIM):
    vb = np.array([int(B_mat[i,j]) for j in range(M)], dtype=np.int64)
    ub_raw = np.array([int(B_mat[i,j]) for j in range(M, DIM)], dtype=np.int64)
    if np.all(vb == 0): continue
    for k in range(-5, 6):
        if k == 0: continue
        v_sol = k * vb
        u_sol_raw = (k * ub_raw + t) % Q
        u_sol = centered(u_sol_raw)
        lv_i, lu_i = int(np.max(np.abs(v_sol))), int(np.max(np.abs(u_sol)))
        si = max(lv_i, lu_i)
        if si < best_s2:
            best_s2 = si
            best_info = (i, k, lv_i, lu_i, v_sol.copy(), u_sol.copy())
        if si <= 15:
            print(f"  *** row={i}, k={k}: score={si} (lv={lv_i}, lu={lu_i})")
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(BASE, 'results_p2', f'solution_p2_{ts}.txt')
            with open(out, 'w') as f:
                f.write(f"v = {v_sol.tolist()}\nu = {u_sol.tolist()}\n")
                f.write(f"# ||v||_inf = {lv_i}\n# ||u||_inf = {lu_i}\n")
                f.write(f"# method: BKZ80-row{i}-k{k}\n")
            print(f"  *** SOLUTION! {out} ***")
print(f"Best inhomogeneous: row={best_info[0]}, k={best_info[1]}, score={best_s2} (lv={best_info[2]}, lu={best_info[3]})")
