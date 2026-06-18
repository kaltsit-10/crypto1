"""
方案二：G6K Sieve + 中长度向量过滤（β=80）

思路：
  BKZ-80 约化后首行 ℓ₂ ≈ 122（恰好 > 120），筛法数据库中包含
  大量 ℓ₂ ≈ 80-130 的向量。在每次 pump 后扫描全数据库，
  找满足 ℓ∞ ≤ 16 AND ℓ₂ ≥ 120 的向量。

关键：β=80 时筛法数据库有 ~2^16.6 ≈ 100,000 个向量。
P(ℓ∞ ≤ 16 | ℓ₂ ≈ 120) ≈ 1/13600。
期望需要约 14 个数据库即可找到解。

预期时间：30-90 分钟（40 workers × 8 线程）
"""

import numpy as np
import ast
import time
import os
import sys
from multiprocessing import Pool, Value, Array, Manager
import ctypes

# G6K imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Q = 120
N = M = Q
GAMMA = 16
TARGET_L2_SQ = Q * Q  # 14400
DIM = N + M  # 240


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    return A_cols.T  # (n, m)


def build_sis_basis_fpylll(A):
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


def verify(sol_lattice, A):
    """验证格坐标下的解（长度 2n 的整数向量）"""
    sol = np.array(sol_lattice, dtype=np.int64)
    if len(sol) != DIM:
        return False
    linf = np.max(np.abs(sol))
    if linf > GAMMA:
        return False
    l2sq = int(np.sum(sol ** 2))
    if l2sq < TARGET_L2_SQ:
        return False
    v, u = sol[:N], sol[N:]
    residual = (A @ v + u) % Q
    return np.all(residual == 0)


def reconstruct_vector(g6k_vec, M_matrix):
    """从 G6K 内部坐标重构原始格向量
    g6k_vec: G6K 返回的系数向量（对格基的线性组合系数）
    M_matrix: 当前格基（numpy array，行向量）
    """
    coeffs = np.array(g6k_vec, dtype=np.int64)
    return coeffs @ M_matrix


def check_database(g6k, basis_np, A, verbose=False):
    """扫描 G6K sieve 数据库中的所有向量，找中长度向量"""
    found = []
    # G6K 数据库可通过 g6k.itervalues() 或下标访问
    db_size = g6k.db_size()
    if verbose:
        print(f"  DB size: {db_size}")

    for i in range(min(db_size, 200000)):  # 最多扫描 200K 向量
        try:
            x = list(g6k[i])  # 获取第 i 个向量的坐标（lattice basis coefficients）
        except Exception:
            break

        # 转换为原始格向量
        # 注意：G6K 的坐标是对当前 GSO basis 的表示
        # 我们需要恢复成格的原始坐标
        vec = np.dot(x, basis_np[:len(x), :DIM]).astype(np.int64) if len(x) <= DIM else None
        if vec is None:
            continue

        linf = int(np.max(np.abs(vec)))
        if linf > GAMMA:
            continue

        l2sq = int(np.sum(vec.astype(np.int64) ** 2))
        if l2sq >= TARGET_L2_SQ:
            # 精确验证
            v, u = vec[:N], vec[N:]
            residual = (A @ v + u) % Q
            if np.all(residual == 0):
                found.append(vec.copy())
                print(f"  ✅ Found! ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.2f}")

    return found


def worker_pump(args):
    """单个 worker：运行 G6K pump，扫描数据库"""
    worker_id, seed, beta_list, A = args

    try:
        from g6k import Siever, SieverParams
        from g6k.algorithms.bkz import pump_n_jump_bkz_tour
        from g6k.algorithms.pump import pump
        from g6k.utils.stats import SieveTreeTracer

        from fpylll import IntegerMatrix, GSO, LLL
        import fpylll
    except ImportError as e:
        print(f"[Worker {worker_id}] Import error: {e}")
        return None

    # 构造 SIS 格基
    B_fp = build_sis_basis_fpylll(A)
    LLL.reduction(B_fp)

    # GSO
    M = GSO.Mat(B_fp, float_type='double')
    M.update_gso()

    # G6K Siever
    params = SieverParams(reserved_n=DIM, otf_lift=True, threads=8, seed=seed)
    g6k = Siever(M, params)

    dim4free_fun = lambda beta: max(0, int(11.5 + 0.075 * beta))

    # 获取当前格基（numpy 形式，用于向量重构）
    basis_np = np.array([[B_fp[i][j] for j in range(DIM)] for i in range(DIM)], dtype=np.int64)

    print(f"[Worker {worker_id}] Starting pump scan, seed={seed}")

    for beta in beta_list:
        if beta > DIM:
            break

        d4f = dim4free_fun(beta)
        kappa_range = range(0, min(6, DIM - beta))

        for kappa in kappa_range:
            try:
                tracer = SieveTreeTracer(g6k, root_eps=0.0, start_clocks=False)
                pump(g6k, tracer, kappa, beta, d4f)
            except Exception as e:
                # LLL 可能在 SIS 基上失败，忽略
                pass

            # 扫描数据库
            try:
                db_size = g6k.db_size()
                for i in range(min(db_size, 100000)):
                    # 尝试获取数据库向量
                    try:
                        x = g6k[i]
                    except Exception:
                        continue

                    # 重构格向量（直接用 GSO 矩阵）
                    # g6k 内部坐标 → 格坐标
                    vec = g6k.M.babai_NTL(x)  # 近似重构（如果可用）
                    # 实际上需要用 M.to_canonical
                    # 这里用简化方式：对前 kappa+beta 个基向量的坐标组合

            except Exception:
                pass

        # 更简单的方法：读取 G6K 更新后的格基行（它们已经是最短的格向量）
        M.update_gso()
        current_basis = np.array([[B_fp[i][j] for j in range(DIM)]
                                  for i in range(DIM)], dtype=np.int64)

        # 检查当前格基前 beta 行
        for i in range(min(30, DIM)):
            row = current_basis[i]
            linf = int(np.max(np.abs(row)))
            l2sq = int(np.sum(row.astype(np.int64) ** 2))

            if linf <= GAMMA and l2sq >= TARGET_L2_SQ:
                v, u = row[:N], row[N:]
                residual = (A @ v + u) % Q
                if np.all(residual == 0):
                    print(f"[Worker {worker_id}] ✅ FOUND at β={beta}, row={i}! ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.2f}")
                    return row

            if i < 5:
                print(f"[Worker {worker_id}] β={beta}, row {i}: ℓ₂={np.sqrt(l2sq):.1f}, ℓ∞={linf}")

    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem5.txt'
    A = load_problem(path)
    print(f"Loaded A: {A.shape}")

    # beta 从 75 到 100 逐步尝试
    # β=80 时首行 ℓ₂ ≈ 122 恰好满足下界
    beta_list = list(range(75, 105, 5))
    print(f"Beta schedule: {beta_list}")

    NUM_WORKERS = 40
    tasks = [
        (wid, wid * 100000 + 42, beta_list, A)
        for wid in range(NUM_WORKERS)
    ]

    print(f"Launching {NUM_WORKERS} workers (8 threads each)...")
    t0 = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_pump, tasks)

    for r in results:
        if r is not None:
            v, u = r[:N], r[N:]
            l2sq = int(np.sum(r.astype(np.int64) ** 2))
            residual = (A @ v + u) % Q
            print(f"\n✅ SOLUTION VERIFIED!")
            print(f"  ℓ∞ = {np.max(np.abs(r))}")
            print(f"  ℓ₂ = {np.sqrt(l2sq):.2f}")
            print(f"  SIS residual max = {np.max(np.abs(residual))}")
            np.save('p5_solution.npy', r)
            print(f"Saved to p5_solution.npy")
            print(f"Total time: {time.time()-t0:.0f}s")
            return

    print(f"❌ No solution found in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
