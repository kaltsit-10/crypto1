#!/usr/bin/env python3
"""P6: Homogeneous SIS — A·v + u ≡ 0 (mod 140), find linf ≤ 17.
n=m=q=140, N=280, gamma=17. Progressive fpylll BKZ + enumeration."""
import numpy as np, ast, os, time, sys
from fpylll import IntegerMatrix, GSO, LLL, BKZ

Q, M, N = 140, 140, 280
GAMMA = 17
HALF = Q // 2  # 70
BASE = '/home/linux/PycharmProjects/pythonProject/crypto1'
P6_DIR = os.path.join(BASE, 'p6优化基')

# Load A matrix
with open(os.path.join(P6_DIR, 'p6_A_matrix.txt')) as f:
    A_list = ast.literal_eval(f.readlines()[-1])
A = np.array(A_list, dtype=np.int64)  # 140x140

# Load existing BKZ-40 basis
with open(os.path.join(P6_DIR, 'p6_basis_bkz40.txt')) as f:
    lines = [l for l in f.readlines() if not l.startswith('#')]
dims = list(map(int, lines[0].split()))
assert dims == [N, N]
B = np.zeros((N, N), dtype=np.int64)
for i in range(N):
    B[i] = np.array(list(map(int, lines[i+1].split())), dtype=np.int64)

# Check current best linf
def linf_of_row(row):
    v = row[:M]
    u_test = row[M:]
    # Center u
    u_c = np.where(u_test % Q <= HALF, u_test % Q, (u_test % Q) - Q)
    v_c = np.where(v % Q <= HALF, v % Q, (v % Q) - Q)
    return max(int(np.abs(v_c).max()), int(np.abs(u_c).max()))

best_li = 999
best_row = -1
for i in range(N):
    if np.all(B[i] == 0):
        continue  # skip trivial zero row
    li = linf_of_row(B[i])
    if li < best_li:
        best_li = li
        best_row = i
print(f"Loaded BKZ-40 basis, best linf={best_li} at row {best_row}")

# Convert to IntegerMatrix
M_fplll = IntegerMatrix(N, N)
for i in range(N):
    for j in range(N):
        M_fplll[i, j] = int(B[i, j])

# Progressive BKZ
start_bs = 45
for bs in range(start_bs, 85, 5):
    if best_li <= GAMMA:
        break
    print(f"BKZ-{bs}...", flush=True)
    t0 = time.time()
    BKZ.reduction(M_fplll, BKZ.EasyParam(bs, max_loops=8))
    dt = time.time() - t0

    # Check new best
    B_cur = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            B_cur[i, j] = int(M_fplll[i, j])

    new_best = 999
    for i in range(N):
        if np.all(B_cur[i] == 0):
            continue
        li = linf_of_row(B_cur[i])
        if li < new_best:
            new_best = li
    print(f"  {dt:.0f}s, best linf={new_best}")

    if new_best < best_li:
        best_li = new_best
        np.save(os.path.join(BASE, f'p6_saves/bkz{bs}_basis.npy'), B_cur)

        # Save solution if found
        if best_li <= GAMMA:
            bv = B_cur[best_row]
            v = bv[:M]; u = bv[M:]
            u_c = np.where(u % Q <= HALF, u % Q, (u % Q) - Q)
            v_c = np.where(v % Q <= HALF, v % Q, (v % Q) - Q)
            np.save(os.path.join(BASE, 'p6_solution.npy'), np.concatenate([v_c, u_c]))
            print(f"*** P6 SOLVED! linf={best_li} ***")
            break

print(f"BKZ done, best linf={best_li}")

# If not solved, enumeration
if best_li > GAMMA:
    print(f"\nEnumeration phase...")
    # Get current basis
    B_cur = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            B_cur[i, j] = int(M_fplll[i, j])

    # Collect short v-vectors
    from itertools import combinations, product
    v_vectors = []
    for i in range(N):
        v_raw = B_cur[i, :M]
        if np.all(v_raw == 0):
            continue
        v_vectors.append(v_raw.copy())
    print(f"  {len(v_vectors)} non-zero v-vectors")

    for depth, topk, crange in [(2, 40, [-3,-2,-1,1,2,3]), (3, 25, [-2,-1,1,2]), (4, 15, [-1,1])]:
        if best_li <= GAMMA:
            break
        print(f"  depth={depth}, top {topk} vectors, coeffs {crange}")
        cnt = 0
        for indices in combinations(range(min(topk, len(v_vectors))), depth):
            vs = [v_vectors[i] for i in indices]
            for cs in product(crange, repeat=depth):
                cnt += 1
                if cnt % 1000000 == 0:
                    print(f"    {cnt/1e6:.0f}M combos, best={best_li}")
                # Compute v = sum(c_i * v_i)
                v_raw = np.zeros(M, dtype=np.int64)
                for d in range(depth):
                    v_raw += cs[d] * vs[d]
                # Center v
                v_c = np.where(v_raw % Q <= HALF, v_raw % Q, (v_raw % Q) - Q)
                ni_v = int(np.abs(v_c).max())
                if ni_v >= best_li:
                    continue
                # Compute u from u = -A·v (mod q)
                u_raw = -A @ v_raw
                u_c = np.where(u_raw % Q <= HALF, u_raw % Q, (u_raw % Q) - Q)
                ni_u = int(np.abs(u_c).max())
                ni = max(ni_v, ni_u)
                if ni < best_li:
                    best_li = ni
                    if ni <= GAMMA:
                        np.save(os.path.join(BASE, 'p6_solution.npy'), np.concatenate([v_c, u_c]))
                        print(f"    *** P6 SOLVED! linf={ni} ***")
                        break
            if best_li <= GAMMA:
                break
        print(f"    {cnt} combos, best={best_li}")

print(f"\nFinal: best linf={best_li}, target={GAMMA}")
