#!/usr/bin/env python3
"""P4: GPU-batched Klein — CPU generates candidates, GPU evaluates all at once."""
import numpy as np, ast, os, time, sys, torch

Q, M_n, DIM = 120, 120, 240; GAMMA = 16

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

# GSO
gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_b[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0: gso_b[i] -= np.dot(Bf[i], gso_b[j]) / gso_r[j] * gso_b[j]
    gso_r[i] = np.dot(gso_b[i], gso_b[i])
target = np.zeros(DIM, dtype=np.float64); target[M_n:] = (-t_vec).astype(np.float64)

# GPU tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{label}] Device: {device}")
B_gpu = torch.from_numpy(Bf).float().to(device)
A_gpu_f = torch.from_numpy(A_mat.astype(np.float32)).float().to(device)
t_gpu = torch.from_numpy(t_vec).long().to(device)

BATCH = 5000

def cpu_klein_batch(rng, n):
    c = np.zeros((n, DIM), dtype=np.int64)
    for b in range(n):
        t_rem = target.copy()
        for j in range(DIM-1, -1, -1):
            if gso_r[j] > 0:
                ex = np.dot(t_rem, gso_b[j]) / gso_r[j]
                if j >= 200: v = int(round(ex + rng.normal(0, 1.0)))
                elif j >= 180: v = int(round(ex + rng.normal(0, 0.7)))
                elif j >= 160: v = int(round(ex + rng.normal(0, 0.5)))
                elif j >= 120: v = int(round(ex + rng.normal(0, 0.3)))
                else: v = int(round(ex))
                c[b,j] = v
                t_rem -= float(v) * Bf[j]
    return c

def gpu_eval(coeffs_np):
    cg = torch.from_numpy(coeffs_np).float().to(device)
    vec = cg @ B_gpu  # (B, 240)
    vg = torch.round(vec[:, :M_n]).long()
    ul = torch.round(vec[:, M_n:]).long()
    u = torch.where((t_gpu.unsqueeze(0) + ul) % Q >= 60,
                    (t_gpu.unsqueeze(0) + ul) % Q - Q,
                    (t_gpu.unsqueeze(0) + ul) % Q)
    Av = (A_gpu_f @ vg.float().T).T.round().long()
    ok = ((Av + u) % Q == t_gpu.unsqueeze(0) % Q).all(dim=1)
    li = torch.max(vg.abs().max(dim=1).values, u.abs().max(dim=1).values)
    li = torch.where(ok, li, torch.tensor(999, device=device))
    return li.cpu().numpy(), vg.cpu().numpy(), u.cpu().numpy()

best = 999
if os.path.exists(save_path):
    try:
        old = np.load(save_path)
        best = max(int(np.max(np.abs(old[:M_n]))), int(np.max(np.abs(old[M_n:]))))
    except: pass

print(f"[{label}] GPU Klein BATCH={BATCH} seed={seed} best={best}")
rng = np.random.RandomState(seed); t0 = time.time(); total = 0

while True:
    coeffs = cpu_klein_batch(rng, BATCH)
    linf_arr, v_arr, u_arr = gpu_eval(coeffs)
    total += BATCH
    bb = int(linf_arr.min())
    if bb < best:
        best = bb; idx = linf_arr.argmin(); dt = time.time()-t0
        print(f"[{label}] t{total} linf={best} |v|={np.abs(v_arr[idx]).max()} |u|={np.abs(u_arr[idx]).max()} ({dt:.0f}s)")
        np.save(save_path, np.concatenate([v_arr[idx].astype(np.int64), u_arr[idx].astype(np.int64)]))
        if best <= GAMMA: print("*** SOLVED! ***"); break
    if total % (BATCH*20) == 0:
        dt = time.time()-t0; print(f"[{label}] {total} best={best} {total/dt:.0f}/s")
