#!/usr/bin/env python3
"""P4: Numba-accelerated Klein CVP worker."""
import numpy as np, ast, os, time, sys
from numba import njit

Q, M, DIM = 120, 120, 240
GAMMA = 16

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
save_path = sys.argv[2] if len(sys.argv) > 2 else 'p4_gpu_solution.npy'
label = os.path.basename(save_path).replace('.npy','')

BASE_DIR = '/home/linux/PycharmProjects/pythonProject/crypto1'
P4_DIR = os.path.join(BASE_DIR, 'p4_deliver/p4_deliver')

with open(os.path.join(P4_DIR, 'crypto1_repo/第一题/sis_inf_problems/problem4.txt')) as f:
    lines = f.read().strip().split('\n')
A_mat = np.array(ast.literal_eval(lines[0].split('=',1)[1].strip()), dtype=np.int64)
t_vec = np.array(ast.literal_eval(lines[1].split('=',1)[1].strip()), dtype=np.int64)

B = np.load(os.path.join(P4_DIR, 'p4_bkz100_l21.npy')).astype(np.int64)
Bf = B.astype(np.float64)

# GSO (done once in Python, too complex for numba)
gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM))
mu = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_b[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0:
            mu[i,j] = np.dot(Bf[i], gso_b[j]) / gso_r[j]
            gso_b[i] -= mu[i,j] * gso_b[j]
    gso_r[i] = np.dot(gso_b[i], gso_b[i])

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
    """Single Klein trial. rand_vals provides pre-generated random numbers."""
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

            if j >= 200:
                c = int(round(c_exact + rand_vals[ri]))
                ri += 1
            elif j >= 180:
                c = int(round(c_exact + rand_vals[ri]))
                ri += 1
            elif j >= 160:
                c = int(round(c_exact + rand_vals[ri]))
                ri += 1
            elif j >= 120:
                c = int(round(c_exact + rand_vals[ri]))
                ri += 1
            else:
                c = int(round(c_exact))

            coeffs[j] = c
            # Subtract c * B[j] from target residual
            for k in range(DIM):
                t_rem[k] -= float(c) * B_arr[j, k]

    return coeffs

@njit
def reconstruct_and_check(coeffs, B_arr, A_arr, t_arr):
    """Reconstruct v,u and check linf."""
    vec = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        c = coeffs[j]
        if c != 0:
            for k in range(DIM):
                vec[k] += float(c) * B_arr[j, k]

    # v = round(vec[:M])
    v = np.zeros(M, dtype=np.int64)
    for i in range(M):
        v[i] = int(round(vec[i]))

    # u_lat = round(vec[M:])
    ul = np.zeros(M, dtype=np.int64)
    for i in range(M):
        ul[i] = int(round(vec[M + i]))

    # u = centered((t + ul) % Q)
    u = np.zeros(M, dtype=np.int64)
    for i in range(M):
        val = (t_arr[i] + ul[i]) % Q
        if val >= 60:
            u[i] = val - Q
        else:
            u[i] = val

    # Check: A@v + u ≡ t (mod Q)
    ok = True
    for i in range(M):
        s = 0
        for j in range(M):
            s += A_arr[i, j] * v[j]
        s += u[i]
        if s % Q != t_arr[i] % Q:
            ok = False
            break

    if not ok:
        return 999

    # linf = max(|v|, |u|)
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

# Preload existing best
best = 999
if os.path.exists(save_path):
    try:
        old = np.load(save_path)
        ov = old[:M]; ou = old[M:]
        best = max(int(np.max(np.abs(ov))), int(np.max(np.abs(ou))))
        print(f"[{label}] Resumed: best={best}")
    except:
        pass

rng = np.random.RandomState(seed)
t0 = time.time()
trial = 0

# Pre-generate random arrays for speed
RAND_BUF_SIZE = 100000

while True:
    # Generate random buffer
    rand_buffer = np.zeros((RAND_BUF_SIZE, 4), dtype=np.float64)
    for i in range(RAND_BUF_SIZE):
        rand_buffer[i, 0] = rng.normal(0, 1.0)   # j>=200
        rand_buffer[i, 1] = rng.normal(0, 0.7)   # j>=180
        rand_buffer[i, 2] = rng.normal(0, 0.5)   # j>=160
        rand_buffer[i, 3] = rng.normal(0, 0.3)   # j>=120

    for i in range(RAND_BUF_SIZE):
        trial += 1
        rand_vals = rand_buffer[i]

        coeffs = klein_trial(target_nb, B_nb, gso_r_nb, gso_b_nb, rand_vals)
        li = reconstruct_and_check(coeffs, B_nb, A_nb, t_nb)

        if li < best:
            best = li
            dt = time.time() - t0
            vec = np.zeros(DIM, dtype=np.float64)
            for j in range(DIM):
                if coeffs[j] != 0:
                    vec += float(coeffs[j]) * Bf[j]
            v = np.round(vec[:M]).astype(np.int64)
            ul = np.round(vec[M:]).astype(np.int64)
            u = (t_vec + ul) % Q
            u = np.where(u >= 60, u - Q, u).astype(np.int64)
            print(f"[{label}] t{trial} linf={li} |v|={np.max(np.abs(v))} |u|={np.max(np.abs(u))} ({dt:.0f}s)")
            np.save(save_path, np.concatenate([v, u]))
            if best <= GAMMA:
                print(f"*** SOLVED! ***")
                sys.exit(0)

        if trial % 500000 == 0:
            dt = time.time() - t0
            print(f"[{label}] {trial} best={best} {trial/dt:.0f}/s")
