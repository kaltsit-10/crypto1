"""
P4 diagnostic: verify Klein on 2080Ti server.
Run this on the server and report output.
"""
import numpy as np, ast

P4 = "crypto1_repo/第一题/sis_inf_problems/problem4.txt"
with open(P4) as f: lines = f.read().strip().split('\n')
A = np.array(ast.literal_eval(lines[0].split("=",1)[1].strip()), dtype=np.int64)
t_vec = np.array(ast.literal_eval(lines[1].split("=",1)[1].strip()), dtype=np.int64)
M,N,Q,DIM = 120,120,120,240

# Load your BKZ-100 basis (homo linf=17)
B = np.load("p4_best.npy")
Bf = B.astype(np.float64)

# GSO
gso_r = np.zeros(DIM)
gso_bstar = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_bstar[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0:
            mu = np.dot(Bf[i], gso_bstar[j]) / gso_r[j]
            gso_bstar[i] -= mu * gso_bstar[j]
    gso_r[i] = np.dot(gso_bstar[i], gso_bstar[i])

print("GSO bottom-10:", [f"{gso_r[i]:.3f}" for i in range(DIM-10, DIM)])

# ===== TEST 1: Verify homogeneous rows =====
print("\n=== TEST 1: Homogeneous basis rows ===")
for i in range(5):
    v = B[i,:M]
    u = B[i,M:] % Q
    u = np.where(u >= 60, u - Q, u)
    ok = np.all((A @ v + u) % Q == 0)
    li = max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))
    print(f"  row {i}: linf={li} v=[{v.min()},{v.max()}] u=[{u.min()},{u.max()}] ok={ok}")

# ===== TEST 2: Zero vector CVP score =====
print("\n=== TEST 2: Zero vector as solution ===")
# Zero = lattice point (0,0). Then u = t mod Q centered.
u_zero = t_vec % Q
u_zero = np.where(u_zero >= 60, u_zero - Q, u_zero)
print(f"  zero vec: v linf=0, u linf={int(np.max(np.abs(u_zero)))}")

# ===== TEST 3: Babai CVP with CORRECT formula =====
print("\n=== TEST 3: Full Babai (no randomization) ===")
# target = (0, -t) — lattice point L should be close to this
# L = (v_L, u_L), then P4: u = (t + u_L) % Q centered
target = np.zeros(DIM, dtype=np.float64)
target[M:] = (-t_vec).astype(np.float64)

coeffs = np.zeros(DIM, dtype=np.int64)
t_rem = target.copy()
for j in range(DIM-1, -1, -1):
    if gso_r[j] > 0:
        c = int(round(np.dot(t_rem, gso_bstar[j]) / gso_r[j]))
        coeffs[j] = c
        t_rem -= float(c) * Bf[j]

# Reconstruct
vec = np.zeros(DIM, dtype=np.float64)
for j in range(DIM):
    if coeffs[j] != 0:
        vec += float(coeffs[j]) * Bf[j]

v = np.round(vec[:M]).astype(np.int64)
u_lat = np.round(vec[M:]).astype(np.int64)

# CORRECT formula
u_p4 = (t_vec + u_lat) % Q
u_p4 = np.where(u_p4 >= 60, u_p4 - Q, u_p4).astype(np.int64)
ok = np.all((A @ v + u_p4 - t_vec) % Q == 0)
lv = int(np.max(np.abs(v)))
lu = int(np.max(np.abs(u_p4)))
print(f"  Formula: u=(t+u_lat)%Q centered")
print(f"  linf={max(lv,lu)} v={lv} u={lu} ok={ok}")

# WRONG formula (common bug)
u_wrong = (t_vec - u_lat) % Q
u_wrong = np.where(u_wrong >= 60, u_wrong - Q, u_wrong).astype(np.int64)
ok_w = np.all((A @ v + u_wrong - t_vec) % Q == 0)
lv_w = int(np.max(np.abs(v)))
lu_w = int(np.max(np.abs(u_wrong)))
print(f"\n  Formula: u=(t-u_lat)%Q centered (WRONG)")
print(f"  linf={max(lv_w,lu_w)} v={lv_w} u={lu_w} ok={ok_w}")

# ===== TEST 4: Best homo row as CVP solution =====
print("\n=== TEST 4: Best homo rows as CVP solutions ===")
# A row is a lattice point L. P4: u = (t + u_L) % Q centered
best_hl = 999
for i in range(min(20, DIM)):
    v = B[i,:M].astype(np.int64)
    u_lat = B[i,M:].astype(np.int64)
    u_p4 = (t_vec + u_lat) % Q
    u_p4 = np.where(u_p4 >= 60, u_p4 - Q, u_p4).astype(np.int64)
    ok = np.all((A @ v + u_p4 - t_vec) % Q == 0)
    li = max(int(np.max(np.abs(v))), int(np.max(np.abs(u_p4))))
    if li < best_hl:
        best_hl = li
        print(f"  row {i}: linf={li} v=[{v.min()},{v.max()}] u=[{u_p4.min()},{u_p4.max()}] ok={ok}")

# ===== TEST 5: Klein 1000 samples =====
print("\n=== TEST 5: Klein sampling (1000) ===")
rng = np.random.RandomState(42)
best = 999
best_v = best_u = 999
for trial in range(1000):
    coeffs = np.zeros(DIM, dtype=np.int64)
    t_rem = target.copy()
    for j in range(DIM-1, -1, -1):
        if gso_r[j] > 0:
            c_exact = np.dot(t_rem, gso_bstar[j]) / gso_r[j]
            if j >= 160:
                c = int(round(c_exact + rng.normal(0, 0.5)))
            else:
                c = int(round(c_exact))
            coeffs[j] = c
            t_rem -= float(c) * Bf[j]

    vec = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        if coeffs[j] != 0:
            vec += float(coeffs[j]) * Bf[j]

    v = np.round(vec[:M]).astype(np.int64)
    u_lat = np.round(vec[M:]).astype(np.int64)
    u_p4 = (t_vec + u_lat) % Q  # CORRECT formula
    u_p4 = np.where(u_p4 >= 60, u_p4 - Q, u_p4).astype(np.int64)
    ok = np.all((A @ v + u_p4 - t_vec) % Q == 0)
    if not ok: continue
    li = max(int(np.max(np.abs(v))), int(np.max(np.abs(u_p4))))
    if li < best:
        best = li
        best_v = int(np.max(np.abs(v)))
        best_u = int(np.max(np.abs(u_p4)))

print(f"  Klein best (1000 samples): linf={best} v={best_v} u={best_u}")
print(f"\n  Expected on BKZ-100: linf ~ 20-22")
print(f"  If yours is >> 30: either formula bug or GSO/Klein bug")
