#!/usr/bin/env python3
"""P4 Klein worker — individual save file, no overwrites."""
import numpy as np, ast, os, time, sys

Q, M, DIM = 120, 120, 240
GAMMA = 16

if len(sys.argv) < 3:
    print("Usage: p4_klein_worker.py <seed> <save_path>")
    sys.exit(1)

seed = int(sys.argv[1])
save_path = sys.argv[2]
label = os.path.basename(save_path).replace('.npy','')

P4_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p4_deliver/p4_deliver')
with open(os.path.join(P4_DIR, 'crypto1_repo/第一题/sis_inf_problems/problem4.txt')) as f:
    lines = f.read().strip().split('\n')
A = np.array(ast.literal_eval(lines[0].split('=',1)[1].strip()), dtype=np.int64)
t = np.array(ast.literal_eval(lines[1].split('=',1)[1].strip()), dtype=np.int64)

B = np.load(os.path.join(P4_DIR, 'p4_bkz100_l21.npy')).astype(np.int64)
Bf = B.astype(np.float64)

gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_b[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0:
            gso_b[i] -= np.dot(Bf[i], gso_b[j]) / gso_r[j] * gso_b[j]
    gso_r[i] = np.dot(gso_b[i], gso_b[i])

target = np.zeros(DIM, dtype=np.float64); target[M:] = (-t).astype(np.float64)
rng = np.random.RandomState(seed)
best = 999; t0 = time.time(); trial = 0

# Load existing best if any
if os.path.exists(save_path):
    try:
        old = np.load(save_path)
        old_v = old[:M]; old_u = old[M:]
        old_li = max(int(np.max(np.abs(old_v))), int(np.max(np.abs(old_u))))
        if np.all((A @ old_v.astype(np.int64) + old_u.astype(np.int64) - t) % Q == 0):
            best = old_li
            print(f'[{label}] Resumed from save: linf={best}')
    except:
        pass

print(f'[{label}] Seed={seed} Starting (best={best})')

while True:
    trial += 1
    coeffs = np.zeros(DIM, dtype=np.int64); t_rem = target.copy()
    for j in range(DIM-1, -1, -1):
        if gso_r[j] > 0:
            c_exact = np.dot(t_rem, gso_b[j]) / gso_r[j]
            if j >= 200: c = int(round(c_exact + rng.normal(0, 1.0)))
            elif j >= 180: c = int(round(c_exact + rng.normal(0, 0.7)))
            elif j >= 160: c = int(round(c_exact + rng.normal(0, 0.5)))
            elif j >= 120: c = int(round(c_exact + rng.normal(0, 0.3)))
            else: c = int(round(c_exact))
            coeffs[j] = c; t_rem -= float(c) * Bf[j]

    vec = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        if coeffs[j]: vec += float(coeffs[j]) * Bf[j]
    v = np.round(vec[:M]).astype(np.int64); ul = np.round(vec[M:]).astype(np.int64)
    u = (t + ul) % Q; u = np.where(u>=60, u-Q, u).astype(np.int64)
    if not np.all((A @ v + u - t) % Q == 0): continue

    li = max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))
    if li < best:
        best = li; dt = time.time()-t0
        np.save(save_path, np.concatenate([v, u]))
        print(f'[{label}] t{trial} linf={li} |v|={np.max(np.abs(v))} |u|={np.max(np.abs(u))} ({dt:.0f}s) SAVED')
        if best <= GAMMA:
            print(f'*** P4 SOLVED by {label}! ***')
            break
    if trial % 100000 == 0:
        print(f'[{label}] {trial} best={best} {trial/(time.time()-t0):.0f}/s')
