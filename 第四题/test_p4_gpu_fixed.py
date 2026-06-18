#!/usr/bin/env python3
"""Test P4 GPU BKZ after nth_element + insert_best_lift fixes."""
import numpy as np, os, sys
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor')
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor/g6k/algorithms')
os.environ['LD_LIBRARY_PATH'] = '/home/linux/G6K-GPU-Tensor/kernel:' + os.environ.get('LD_LIBRARY_PATH','')
from fpylll import IntegerMatrix, GSO
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.utils.stats import dummy_tracer

B = np.load('/home/linux/PycharmProjects/pythonProject/crypto1/p4_deliver/p4_deliver/p4_bkz100_l21.npy').astype(np.int64)
N = 240
m = IntegerMatrix(N, N)
for i in range(N):
    for j in range(N): m[i, j] = int(B[i, j])
mg = GSO.Mat(m, U=IntegerMatrix.identity(N), UinvT=IntegerMatrix.identity(N))
mg.update_gso()

p = SieverParams(threads=4)
g = Siever(mg, p, seed=0)
print('P4 GPU BKZ-50 full test (with nth+insert fixes)...', flush=True)

try:
    pump_n_jump_bkz_tour(g, dummy_tracer, 50, pump_params={'down_sieve': False}, verbose=True)
    B2 = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N): B2[i, j] = int(g.M.B[i, j])
    Q, M, HALF = 120, 120, 60
    best = 999
    for i in range(N):
        row = B2[i]; v = row[:M]; u = row[M:] % Q
        u = np.where(u >= HALF, u - Q, u).astype(np.int64)
        if np.all(v == 0) and np.all(u == 0): continue
        li = max(int(np.abs(v).max()), int(np.abs(u).max()))
        if li < best: best = li
    print(f'*** GPU BKZ-50 DONE! Best linf={best} ***')
except Exception as e:
    import traceback; traceback.print_exc()
