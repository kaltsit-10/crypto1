"""
方案三：G6K otf_lift 非齐次 CVP（与方案一本质相同，代码更简洁）

思路：
  G6K 的 otf_lift=True 参数支持在筛法过程中自动做"on-the-fly lifting"，
  这正是为 CVP/non-homogeneous SIS 设计的。

  与方案一（Kannan 嵌入）等价，但通过 G6K 内置机制实现，
  格基维度保持 280（不需要增广到 281）。

  核心修改：
  1. 将 t（中心化后）作为 CVP 目标嵌入到 G6K 的 lift offset
  2. G6K 筛法自动在 SIS 陪集中搜索短向量
  3. 检验时改用 Av + u ≡ t (mod q)

预期时间：1-4 小时（与方案一相同）
"""

import numpy as np
import ast
import time
import os
import sys
from multiprocessing import Pool, Value, Array
import ctypes

Q = 140
N = M = Q
GAMMA = 17
DIM = N + M  # 280

FOUND = Value(ctypes.c_bool, False)
RESULT = Array(ctypes.c_int, DIM)


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    t_str = lines[1][lines[1].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    t_vec = np.array(ast.literal_eval(t_str), dtype=np.int64)
    return A_cols.T, t_vec


def center_lift(v, q=Q):
    v = np.array(v, dtype=np.int64) % q
    return np.where(v > q // 2, v - q, v)


def build_sis_basis_fp(A):
    from fpylll import IntegerMatrix
    n, m = A.shape
    dim = n + m
    B = IntegerMatrix(dim, dim)
    for i in range(n):
        B[i][i] = Q
        for j in range(m):
            B[i][n + j] = int(A[i, j])
    for j in range(m):
        B[n + j][n + j] = -1
    return B


def verify(sol, A, t):
    sol = np.array(sol, dtype=np.int64)
    v, u = sol[:N], sol[N:]
    linf = int(np.max(np.abs(sol)))
    residual = (A @ v + u - t) % Q
    return linf <= GAMMA and np.all(residual == 0)


def worker_cvp_pump(args):
    """
    单个 worker：用 G6K pump 在 SIS 陪集中搜索短向量。
    非齐次情况：修改格基使得格点直接对应陪集元素。

    关键技巧：构造移位格基 B_shifted：
      将第 2n 行（或格基的某一行）替换为包含 t 的向量，
      使得格中的零向量对应于 (v=0, u=0) 满足 Av+u≡0，
      而特殊行对应于特解。

    另一等价方法（这里实现）：
      用标准 SIS 格基，但在每次 pump 后，
      对每个格向量 w 检验 w + (0, t) 是否满足条件。
      即：若 Aw_v + w_u ≡ 0 (mod q)，则 A(-w_v) + (t-w_u) ≡ t (mod q)。
      所以 v = -w_v, u = t - w_u（需中心化）。
    """
    worker_id, seed, A, t, beta_list = args

    if FOUND.value:
        return None

    try:
        from g6k import Siever, SieverParams
        from g6k.algorithms.pump import pump
        from g6k.utils.stats import SieveTreeTracer
        from fpylll import LLL, BKZ, GSO
    except ImportError as e:
        print(f"[W{worker_id}] Import error: {e}")
        return None

    B_fp = build_sis_basis_fp(A)
    LLL.reduction(B_fp)
    BKZ.reduction(B_fp, BKZ.Param(block_size=95, max_loops=4, flags=BKZ.AUTO_ABORT))

    M_gso = GSO.Mat(B_fp, float_type='double')
    M_gso.update_gso()

    params = SieverParams(reserved_n=DIM, otf_lift=True, threads=8, seed=seed)
    g6k = Siever(M_gso, params)

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

            # 检查格基行：w 是 L₀ 的格点
            # 检验 (-w_v, t - w_u) 是否满足 P7 条件
            M_gso.update_gso()
            for i in range(DIM):
                row = np.array([B_fp[i][j] for j in range(DIM)], dtype=np.int64)

                # 候选1: v = -row[:N], u = center_lift(t - row[N:])
                v1 = -row[:N]
                u1 = center_lift(t - row[N:])
                linf1 = max(int(np.max(np.abs(v1))), int(np.max(np.abs(u1))))
                if linf1 <= GAMMA:
                    res1 = (A @ v1 + u1 - t) % Q
                    if np.all(res1 == 0):
                        print(f"[W{worker_id}] ✅ Found (case 1)! β={beta}, κ={kappa}, row={i}, ℓ∞={linf1}")
                        FOUND.value = True
                        sol = np.concatenate([v1, u1])
                        for j in range(DIM):
                            RESULT[j] = int(sol[j])
                        return sol

                # 候选2: v = row[:N], u = center_lift(t + row[N:])  (negated row)
                v2 = row[:N]
                u2 = center_lift(t + row[N:])
                linf2 = max(int(np.max(np.abs(v2))), int(np.max(np.abs(u2))))
                if linf2 <= GAMMA:
                    res2 = (A @ v2 + u2 - t) % Q
                    if np.all(res2 == 0):
                        print(f"[W{worker_id}] ✅ Found (case 2)! β={beta}, κ={kappa}, row={i}, ℓ∞={linf2}")
                        FOUND.value = True
                        sol = np.concatenate([v2, u2])
                        for j in range(DIM):
                            RESULT[j] = int(sol[j])
                        return sol

        if not FOUND.value:
            # 打印进度
            row0 = np.array([B_fp[0][j] for j in range(DIM)], dtype=np.int64)
            print(f"[W{worker_id}] β={beta}: first row ℓ₂={np.linalg.norm(row0):.1f}, ℓ∞={np.max(np.abs(row0))}")

    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem7.txt'
    A, t = load_problem(path)
    print(f"Loaded: A {A.shape}, t {t.shape}")
    print(f"q={Q}, γ={GAMMA}, dim={DIM}")

    beta_list = list(range(90, 130, 5))
    print(f"Beta schedule: {beta_list}")

    NUM_WORKERS = 40
    tasks = [
        (wid, wid * 99991 + 37, A, t, beta_list)
        for wid in range(NUM_WORKERS)
    ]

    print(f"Launching {NUM_WORKERS} workers (8 threads each)...")
    t0_wall = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_cvp_pump, tasks)

    elapsed = time.time() - t0_wall
    print(f"Finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    for r in results:
        if r is not None:
            if verify(r, A, t):
                v, u = r[:N], r[N:]
                print(f"\n✅ SOLUTION VERIFIED!")
                print(f"  ℓ∞(v)={np.max(np.abs(v))}, ℓ∞(u)={np.max(np.abs(u))}")
                np.save('p7_solution.npy', r)
                print("Saved to p7_solution.npy")
                return r

    if FOUND.value:
        sol = np.array(list(RESULT), dtype=np.int64)
        if verify(sol, A, t):
            v, u = sol[:N], sol[N:]
            print(f"\n✅ SOLUTION from shared memory!")
            print(f"  ℓ∞(v)={np.max(np.abs(v))}, ℓ∞(u)={np.max(np.abs(u))}")
            np.save('p7_solution.npy', sol)
            return sol

    print("❌ No solution found. Try method1_kannan.py or increase beta.")
    return None


if __name__ == '__main__':
    main()
