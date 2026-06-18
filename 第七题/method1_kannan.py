"""
方案一：Kannan 嵌入 + G6K pump（主力方案，直接改造 P3 代码）

思路：
  构造 281×281 的 Kannan 增广格基，将非齐次 SIS（Av+u≡t）
  转化为同维增广格中的 SVP 问题。
  运行与 P3 完全相同的并行 G6K pump 框架，
  在检验时寻找最后分量为 ±γ 的短格向量。

Kannan 格基结构（281×281）：
  行 0..n-1:  [ q·I_n |  A    |  0  ]
  行 n..2n-1: [  0    | -I_n  |  0  ]
  行 2n:      [  0    |  t^T  |  γ  ]

预期时间：1-4 小时（40 workers × 8 线程）
硬件需求：G6K + fpylll
"""

import numpy as np
import ast
import time
import os
import sys
from multiprocessing import Pool, Value, Array
import ctypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Q = 140
N = M = Q
GAMMA = 17
DIM_SIS = N + M     # 280
DIM_EXT = DIM_SIS + 1  # 281 (Kannan extended)

FOUND = Value(ctypes.c_bool, False)
RESULT_V = Array(ctypes.c_int, N)
RESULT_U = Array(ctypes.c_int, N)


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    t_str = lines[1][lines[1].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    t = np.array(ast.literal_eval(t_str), dtype=np.int64)
    return A_cols.T, t


def center_lift(v, q):
    v = np.array(v, dtype=np.int64) % q
    return np.where(v > q // 2, v - q, v)


def build_kannan_basis_fp(A, t):
    """构造 281×281 Kannan 增广格基（fpylll IntegerMatrix）

    列索引:  0..n-1   (v 部分)
             n..2n-1  (u 部分)
             2n       (Kannan extra 列)

    行 i (0≤i<n):   q·e_i 在 v 部分，A 的第 i 行在 u 部分，0 在最后列
    行 n+j (0≤j<n): 0 在 v 部分，-e_j 在 u 部分，0 在最后列
    行 2n:           0 在 v 部分，t 在 u 部分，γ 在最后列
    """
    from fpylll import IntegerMatrix

    dim = DIM_EXT  # 281
    B = IntegerMatrix(dim, dim)

    # 行 0..n-1: [q·I | A | 0]
    for i in range(N):
        B[i][i] = Q
        for j in range(M):
            B[i][N + j] = int(A[i, j])
        # B[i][DIM_SIS] = 0 (already)

    # 行 n..2n-1: [0 | -I | 0]
    for j in range(M):
        B[N + j][N + j] = -1

    # 行 2n: [0 | t | γ]
    for j in range(N):
        B[DIM_SIS][N + j] = int(t[j])
    B[DIM_SIS][DIM_SIS] = GAMMA

    return B


def verify(v, u, A, t):
    v = np.array(v, dtype=np.int64)
    u = np.array(u, dtype=np.int64)
    linf = max(int(np.max(np.abs(v))), int(np.max(np.abs(u))))
    residual = (A @ v + u - t) % Q
    return linf <= GAMMA and np.all(residual == 0)


def check_kannan_row(row, A, t):
    """
    检验 Kannan 格基行是否对应 P7 的解。
    row: 长度 281 的整数向量
    最后分量应为 ±γ，前 2n 分量给出 (v, u-t)（或其负）。
    """
    last = int(row[DIM_SIS])
    if abs(last) != GAMMA:
        return None

    sign = 1 if last > 0 else -1
    v = sign * row[:N]
    u_minus_t_scaled = sign * row[N:DIM_SIS]

    # 恢复 u：u = u_minus_t + t（需要模 q 中心化）
    # row 中 u 部分编码的是 u - t（缩放后），直接还原
    u = center_lift(u_minus_t_scaled + t, Q)

    if verify(v, u, A, t):
        return v, u
    return None


def worker_pump(args):
    worker_id, seed, A, t, beta_list = args

    if FOUND.value:
        return None

    try:
        from g6k import Siever, SieverParams
        from g6k.algorithms.pump import pump
        from g6k.utils.stats import SieveTreeTracer
        from fpylll import LLL, BKZ
    except ImportError as e:
        print(f"[W{worker_id}] Import error: {e}")
        return None

    B_fp = build_kannan_basis_fp(A, t)
    LLL.reduction(B_fp)
    BKZ.reduction(B_fp, BKZ.Param(block_size=95, max_loops=4, flags=BKZ.AUTO_ABORT))

    from fpylll import GSO
    M = GSO.Mat(B_fp, float_type='double')
    M.update_gso()

    from g6k import SieverParams
    params = SieverParams(reserved_n=DIM_EXT, otf_lift=True, threads=8, seed=seed)
    g6k = Siever(M, params)

    def d4f(beta): return max(0, int(11.5 + 0.075 * beta))

    for beta in beta_list:
        if FOUND.value:
            return None

        for kappa in range(0, 6):
            if FOUND.value:
                return None
            try:
                tracer = SieveTreeTracer(g6k, root_eps=0.0, start_clocks=False)
                pump(g6k, tracer, kappa, beta, d4f(beta))
            except Exception:
                pass

            # 检查当前格基所有行
            M.update_gso()
            for i in range(DIM_EXT):
                row = np.array([B_fp[i][j] for j in range(DIM_EXT)], dtype=np.int64)
                result = check_kannan_row(row, A, t)
                if result is not None:
                    v, u = result
                    print(f"[W{worker_id}] ✅ FOUND! β={beta}, κ={kappa}, row={i}")
                    print(f"  ℓ∞={max(np.max(np.abs(v)), np.max(np.abs(u)))}")
                    FOUND.value = True
                    for j in range(N):
                        RESULT_V[j] = int(v[j])
                        RESULT_U[j] = int(u[j])
                    return (np.array(v), np.array(u))

        print(f"[W{worker_id}] β={beta} done, no solution yet")

    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem7.txt'
    A, t = load_problem(path)
    print(f"Loaded: A {A.shape}, t {t.shape}")
    print(f"q={Q}, γ={GAMMA}")
    print(f"Kannan extended dim = {DIM_EXT}")

    beta_list = list(range(90, 130, 5))
    print(f"Beta schedule: {beta_list}")

    NUM_WORKERS = 40
    tasks = [
        (wid, wid * 99991 + 17, A, t, beta_list)
        for wid in range(NUM_WORKERS)
    ]

    print(f"\nLaunching {NUM_WORKERS} workers (8 threads each)...")
    t0 = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_pump, tasks)

    elapsed = time.time() - t0
    print(f"\nSearch finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # 检查结果
    for r in results:
        if r is not None:
            v, u = r
            if verify(v, u, A, t):
                print(f"✅ SOLUTION VERIFIED!")
                print(f"  ℓ∞(v)={np.max(np.abs(v))}, ℓ∞(u)={np.max(np.abs(u))}")
                sol = np.concatenate([v, u])
                np.save('p7_solution.npy', sol)
                print("Saved to p7_solution.npy")
                return sol

    # 检查 shared memory
    if FOUND.value:
        v = np.array(list(RESULT_V), dtype=np.int64)
        u = np.array(list(RESULT_U), dtype=np.int64)
        if verify(v, u, A, t):
            print("✅ SOLUTION from shared memory!")
            sol = np.concatenate([v, u])
            np.save('p7_solution.npy', sol)
            return sol

    print("❌ No solution found. Try increasing beta range or workers.")
    return None


if __name__ == '__main__':
    main()
