"""
方案一：BKZ-60 预约化 + 并行随机格基行组合搜索

思路：
  BKZ-60 约化后前几个格基行有 ℓ₂ ≈ 120-159，已在目标范围。
  对这些行做 {-2,-1,0,1,2} 系数组合，并行搜索满足 ℓ∞ ≤ 16 的组合。
  全部格基行的整数线性组合都是有效 SIS 格点。

预期时间：5-30 分钟（190核）
硬件需求：纯 CPU + NumPy，不需要 G6K
"""

import numpy as np
import ast
from fpylll import IntegerMatrix, LLL, BKZ
from multiprocessing import Pool, Value, Array
import ctypes
import itertools
import time

Q = 120
N = M = Q
GAMMA = 16
TARGET_L2_SQ = Q * Q  # 14400

FOUND = Value(ctypes.c_bool, False)
RESULT = Array(ctypes.c_int, 2 * N)


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    return A_cols.T  # shape (n, m)


def build_sis_basis(A):
    """构造 SIS 格基 B: (2n)×(2n) 整数矩阵
    前n行: [q*I_n | A]   (A 作为列向量块)
    后n行: [0    | -I_n]
    每行 = 一个格向量
    """
    n, m = A.shape
    dim = n + m
    B = np.zeros((dim, dim), dtype=np.int64)
    B[:n, :n] = Q * np.eye(n, dtype=np.int64)
    B[:n, n:] = A  # A is (n,m), n=m=120
    B[n:, n:] = -np.eye(m, dtype=np.int64)
    return B


def to_fpylll(B_np):
    """numpy int64 矩阵 → fpylll IntegerMatrix"""
    n = B_np.shape[0]
    B = IntegerMatrix(n, n)
    for i in range(n):
        for j in range(n):
            B[i][j] = int(B_np[i, j])
    return B


def from_fpylll(B_fp):
    """fpylll IntegerMatrix → numpy int64"""
    n = B_fp.nrows
    return np.array([[B_fp[i][j] for j in range(n)] for i in range(n)], dtype=np.int64)


def verify(sol, A):
    v, u = sol[:N], sol[N:]
    if np.max(np.abs(sol)) > GAMMA:
        return False
    l2sq = int(np.sum(sol.astype(np.int64) ** 2))
    if l2sq < TARGET_L2_SQ:
        return False
    residual = (A @ v + u) % Q
    return np.all(residual == 0)


def run_bkz60(A):
    """构造 SIS 格基并运行 BKZ-60，返回 numpy 格基"""
    print("[Main] Building SIS lattice basis (240×240)...")
    B_np = build_sis_basis(A)
    B_fp = to_fpylll(B_np)

    print("[Main] Running LLL...")
    LLL.reduction(B_fp)

    print("[Main] Running BKZ-60 (expect 15-40 min)...")
    t0 = time.time()
    BKZ.reduction(B_fp, BKZ.Param(
        block_size=60,
        max_loops=8,
        flags=BKZ.VERBOSE | BKZ.AUTO_ABORT
    ))
    print(f"[Main] BKZ-60 done in {time.time()-t0:.0f}s")

    B_reduced = from_fpylll(B_fp)

    # 打印前20行的 ℓ₂
    print("[Main] Top 20 row norms after BKZ-60:")
    for i in range(20):
        row = B_reduced[i]
        l2 = np.sqrt(np.sum(row.astype(np.float64)**2))
        linf = np.max(np.abs(row))
        print(f"  row {i:2d}: ℓ₂={l2:.1f}, ℓ∞={linf}")

    return B_reduced


def worker_search(args):
    """单个 worker：在格基前 K 行的组合空间中搜索"""
    worker_id, B_rows, seed, n_iters = args
    rng = np.random.default_rng(seed)
    K = B_rows.shape[0]

    # 批处理：一次生成 BATCH 个系数向量
    BATCH = 2000
    found_count = 0

    for _ in range(n_iters // BATCH):
        if FOUND.value:
            return None

        # 系数：整数，范围 [-2, 2]，shape (BATCH, K)
        coeffs = rng.integers(-2, 3, size=(BATCH, K))

        # 批量计算组合向量：shape (BATCH, 240)
        vecs = coeffs @ B_rows  # (BATCH, 240)

        # 检查 ℓ∞
        linf = np.max(np.abs(vecs), axis=1)  # (BATCH,)
        mask_inf = linf <= GAMMA

        # 检查 ℓ₂
        l2sq = np.sum(vecs.astype(np.float64) ** 2, axis=1)
        mask_l2 = l2sq >= TARGET_L2_SQ

        mask = mask_inf & mask_l2
        if np.any(mask):
            idx = np.where(mask)[0][0]
            sol = vecs[idx]
            # 精确验证（防止整数溢出导致的假阳性）
            if np.max(np.abs(sol)) <= GAMMA and int(np.sum(sol.astype(np.int64)**2)) >= TARGET_L2_SQ:
                FOUND.value = True
                sol_int = sol.astype(np.int64)
                for j in range(len(sol_int)):
                    RESULT[j] = int(sol_int[j])
                print(f"[Worker {worker_id}] FOUND! ℓ∞={np.max(np.abs(sol))}, ℓ₂={np.sqrt(np.sum(sol**2)):.2f}")
                return sol_int

        found_count += BATCH

    return None


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem5.txt'

    A = load_problem(path)
    print(f"Loaded A: {A.shape}, q={Q}, γ={GAMMA}")

    # Step 1: BKZ-60
    bkz_save = '/tmp/p5_bkz60_basis.npy'
    try:
        B_reduced = np.load(bkz_save)
        print(f"[Main] Loaded saved BKZ-60 basis from {bkz_save}")
    except FileNotFoundError:
        B_reduced = run_bkz60(A)
        np.save(bkz_save, B_reduced)
        print(f"[Main] Saved BKZ-60 basis to {bkz_save}")

    # Step 2: 挑选 ℓ₂ 在 [120, 250] 的前几行
    target_rows = []
    for i in range(60):  # 检查前60行
        row = B_reduced[i]
        l2 = np.sqrt(float(np.sum(row.astype(np.float64)**2)))
        if TARGET_L2_SQ**0.5 * 0.8 <= l2 <= GAMMA * np.sqrt(2*N):
            target_rows.append(row)
            print(f"  Selected row {i}: ℓ₂={l2:.1f}")

    if not target_rows:
        print("[Main] No rows in target range. Using all first 30 rows.")
        target_rows = [B_reduced[i] for i in range(30)]

    B_rows = np.array(target_rows, dtype=np.int64)
    K = len(B_rows)
    print(f"[Main] Using {K} basis rows for combination search")

    # 直接检验每个基向量本身
    print("[Main] Checking basis rows directly...")
    for i, row in enumerate(B_rows):
        linf = np.max(np.abs(row))
        l2sq = int(np.sum(row.astype(np.int64)**2))
        print(f"  row {i}: ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.1f}")
        if linf <= GAMMA and l2sq >= TARGET_L2_SQ:
            v, u = row[:N], row[N:]
            if verify(row, A):
                print(f"[Main] ✅ Direct basis row {i} is a solution!")
                np.save('p5_solution.npy', row)
                return row

    # Step 3: 并行搜索
    NUM_WORKERS = 190
    N_ITERS_PER_WORKER = 5_000_000  # 每个 worker 尝试 500 万次

    print(f"[Main] Launching {NUM_WORKERS} parallel workers, {N_ITERS_PER_WORKER:,} iters each...")
    print(f"[Main] Total: {NUM_WORKERS * N_ITERS_PER_WORKER:,} combinations to try")
    print(f"[Main] Expected success probability: ~{NUM_WORKERS * N_ITERS_PER_WORKER / 1_300_000:.0f}x oversampling")

    t0 = time.time()
    tasks = [
        (wid, B_rows, wid * 123456 + 7, N_ITERS_PER_WORKER)
        for wid in range(NUM_WORKERS)
    ]

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_search, tasks)

    elapsed = time.time() - t0
    print(f"[Main] Search finished in {elapsed:.1f}s")

    # 检查结果
    for r in results:
        if r is not None:
            v, u = r[:N], r[N:]
            if verify(r, A):
                print(f"✅ Solution found! ℓ∞={np.max(np.abs(r))}, ℓ₂={np.sqrt(np.sum(r**2)):.2f}")
                np.save('p5_solution.npy', r)
                print(f"Saved to p5_solution.npy")
                return r

    # 检查 shared memory
    if FOUND.value:
        sol = np.array(list(RESULT), dtype=np.int64)
        if verify(sol, A):
            print(f"✅ Solution found via shared memory!")
            np.save('p5_solution.npy', sol)
            return sol

    print("❌ No solution found. Try increasing N_ITERS_PER_WORKER or K.")
    return None


if __name__ == '__main__':
    main()
