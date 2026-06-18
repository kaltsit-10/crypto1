#!/usr/bin/env python3
"""P4: Klein + enumeration on BKZ-110 (homo linf=16) basis. Enumerate top dims where GS is large."""
import numpy as np, ast, os, time, itertools, sys

Q, M, DIM = 120, 120, 240; GAMMA = 16
BASE = '/home/linux/PycharmProjects/pythonProject/crypto1'
P4_DIR = os.path.join(BASE, 'p4_deliver/p4_deliver')

with open(os.path.join(P4_DIR, 'crypto1_repo/第一题/sis_inf_problems/problem4.txt')) as f:
    lines = f.read().strip().split('\n')
A = np.array(ast.literal_eval(lines[0].split('=',1)[1].strip()), dtype=np.int64)
t = np.array(ast.literal_eval(lines[1].split('=',1)[1].strip()), dtype=np.int64)

B = np.load(os.path.join(BASE, 'p4_gpu_best.npy')).astype(np.int64)
Bf = B.astype(np.float64)

# GSO
gso_r = np.zeros(DIM); gso_b = np.zeros((DIM,DIM)); mu = np.zeros((DIM,DIM))
for i in range(DIM):
    gso_b[i] = Bf[i].copy()
    for j in range(i):
        if gso_r[j] > 0:
            mu[i,j] = np.dot(Bf[i], gso_b[j]) / gso_r[j]
            gso_b[i] -= mu[i,j] * gso_b[j]
    gso_r[i] = np.dot(gso_b[i], gso_b[i])

target = np.zeros(DIM, dtype=np.float64); target[M:] = (-t).astype(np.float64)
print(f"Homo linf=16, GS[0]={gso_r[0]:.0f} GS[50]={gso_r[50]:.0f} GS[100]={gso_r[100]:.0f}")

# Strategy: Klein on dims 80-239, enumerate dims 0-79 with pruning
ENUM_START = 80  # Enumerate dims 0 to ENUM_START-1
ENUM_RANGE = 1   # Try ±1 around Klein rounding

B_v = Bf[:, :M].copy(); B_u = Bf[:, M:].copy()

def compute_linf_from_coeffs(coeffs):
    vec_v = np.zeros(M); vec_u = np.zeros(M)
    for j in range(DIM):
        if coeffs[j]: vec_v += float(coeffs[j]) * B_v[j]; vec_u += float(coeffs[j]) * B_u[j]
    v = np.round(vec_v).astype(np.int64); ul = np.round(vec_u).astype(np.int64)
    u = (t + ul) % Q; u = np.where(u>=60, u-Q, u).astype(np.int64)
    if not np.all((A @ v + u - t) % Q == 0): return 999, None, None
    return max(int(np.max(np.abs(v))), int(np.max(np.abs(u)))), v, u

def klein_top_dims(coeffs_base, rng):
    """Run Klein for dims ENUM_START to DIM-1, return completed coeffs."""
    coeffs = coeffs_base.copy()
    t_rem = target.copy()
    # Subtract contribution of dims 0 to ENUM_START-1
    for j in range(ENUM_START):
        c = coeffs[j]
        if c != 0:
            t_rem -= float(c) * Bf[j]
    # Klein for remaining dims
    for j in range(DIM-1, ENUM_START-1, -1):
        if gso_r[j] > 0:
            ce = np.dot(t_rem, gso_b[j]) / gso_r[j]
            if j >= 200: c = int(round(ce + rng.normal(0, 1.0)))
            elif j >= 180: c = int(round(ce + rng.normal(0, 0.7)))
            elif j >= 160: c = int(round(ce + rng.normal(0, 0.5)))
            elif j >= 120: c = int(round(ce + rng.normal(0, 0.3)))
            else: c = int(round(ce))
            coeffs[j] = c
            t_rem -= float(c) * Bf[j]
    return coeffs

def enumerate_dims(coeffs_template, rng):
    """Enumerate dims 0 to ENUM_START-1 with pruning."""
    # First get baseline (Klein rounding for these dims)
    t_rem = target.copy()
    # Subtract contribution of dims >= ENUM_START
    for j in range(ENUM_START, DIM):
        c = coeffs_template[j]
        if c != 0:
            t_rem -= float(c) * Bf[j]
    # Now compute Babai coeffs for dims 0 to ENUM_START-1
    baseline = np.zeros(ENUM_START, dtype=np.int64)
    for j in range(ENUM_START-1, -1, -1):
        if gso_r[j] > 0:
            ce = np.dot(t_rem, gso_b[j]) / gso_r[j]
            baseline[j] = int(round(ce))
            t_rem -= float(baseline[j]) * Bf[j]
    return baseline

# Main loop
rng = np.random.RandomState(42)
best = 999; best_v = best_u = None; t0 = time.time(); trial = 0
save_path = os.path.join(BASE, 'p4_saves/enum_b16.npy')

# Precompute all ±1 combinations for ENUM_START dims (2^80 too many!)
# Instead: enumerate progressively with beam pruning
# Beam search on first 30 dims (indices 0-29), Klein for rest

BEAM_DIMS = 30  # Enumerate these
BEAM_WIDTH = 200

print(f"Beam search on dims 0-{BEAM_DIMS-1} (GS range: {gso_r[0]:.0f} - {gso_r[BEAM_DIMS-1]:.1f}), width={BEAM_WIDTH}")

while True:
    trial += 1
    # Generate baseline Klein for ALL dims
    coeffs_full = np.zeros(DIM, dtype=np.int64)
    t_rem = target.copy()
    for j in range(DIM-1, -1, -1):
        if gso_r[j] > 0:
            ce = np.dot(t_rem, gso_b[j]) / gso_r[j]
            if j >= 200: c = int(round(ce + rng.normal(0, 1.0)))
            elif j >= 180: c = int(round(ce + rng.normal(0, 0.7)))
            elif j >= 160: c = int(round(ce + rng.normal(0, 0.5)))
            elif j >= 120: c = int(round(ce + rng.normal(0, 0.3)))
            else: c = int(round(ce))
            coeffs_full[j] = c; t_rem -= float(c) * Bf[j]

    li_base, v_base, u_base = compute_linf_from_coeffs(coeffs_full)
    li = li_base; best_c = coeffs_full.copy()

    # Beam search on first BEAM_DIMS to improve
    if li_base < 999:
        # Init beam with baseline
        beams = [(coeffs_full.copy(), li_base)]
        for bdim in range(BEAM_DIMS-1, -1, -1):
            new_beams = []
            for coeffs_b, li_b in beams:
                for d in [-1, 0, 1]:
                    c2 = coeffs_b.copy(); c2[bdim] += d
                    li2, _, _ = compute_linf_from_coeffs(c2)
                    if li2 < 999:
                        new_beams.append((c2, li2))
            # Keep best BEAM_WIDTH by linf
            new_beams.sort(key=lambda x: x[1])
            beams = new_beams[:BEAM_WIDTH]
        # Best from beam
        if beams and beams[0][1] < li:
            li = beams[0][1]
            best_c = beams[0][0]

    if li < best:
        best = li; _, best_v, best_u = compute_linf_from_coeffs(best_c)
        dt = time.time()-t0
        print(f"t{trial} linf={best} |v|={np.max(np.abs(best_v))} |u|={np.max(np.abs(best_u))} (K={li_base}) ({dt:.0f}s)")
        np.save(save_path, np.concatenate([best_v, best_u]))
        if best <= GAMMA: print("*** SOLVED! ***"); break
    if trial % 1000 == 0:
        dt = time.time()-t0; print(f"[{trial}] best={best} {trial/dt:.1f}/s")
