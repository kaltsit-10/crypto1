"""
方案三：P3 代码复用 + 多解收集与后处理

思路：
  P3 的解 ℓ₂ = 109.5，距离 P5 的条件差 10.4。
  核心观察：G6K 筛法找到的"第一个" ℓ∞ ≤ 16 的解不是唯一的，
  不同随机种子和不同 β 路径会找到不同的解，这些解的 ℓ₂ 分布各异。
  收集足够多的 ℓ∞ ≤ 16 解后，其中天然会有 ℓ₂ ≥ 120 的。

理论期望：
  P3 解的 ℓ₂ 约为 109.5，比 120 小约 10%。
  通过调整筛法参数（使用较高 beta 或不同随机种子），
  可以找到 ℓ₂ 分布更"均匀"的解。
  大约尝试 14 次独立运行即可期望找到 ℓ₂ ≥ 120 的解。

另外提供：两解组合策略（找两个 ℓ∞ ≤ 8 的解，相加得到 ℓ∞ ≤ 16 的解）。

预期时间：1-3 小时（与 P3 相同架构）
"""

import numpy as np
import ast
import time
import os
import sys
from multiprocessing import Pool, Manager
from fpylll import IntegerMatrix, GSO, LLL, BKZ
import fpylll

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../第三题'))

Q = 120
N = M = Q
GAMMA = 16
TARGET_L2_SQ = Q * Q
DIM = N + M

# 全局共享：收集到的所有合法解
collected_solutions = []


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    return A_cols.T


def build_sis_basis_fpylll(A):
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


def verify_full(sol, A):
    sol = np.array(sol, dtype=np.int64)
    linf = np.max(np.abs(sol))
    l2sq = int(np.sum(sol ** 2))
    v, u = sol[:N], sol[N:]
    residual = (A @ v + u) % Q
    sis_ok = np.all(residual == 0)
    return linf <= GAMMA and l2sq >= TARGET_L2_SQ and sis_ok


def verify_linf_only(sol, A):
    """只验证 ℓ∞ ≤ γ 和 SIS 方程（不要求 ℓ₂ 下界）"""
    sol = np.array(sol, dtype=np.int64)
    linf = np.max(np.abs(sol))
    if linf > GAMMA:
        return False
    v, u = sol[:N], sol[N:]
    residual = (A @ v + u) % Q
    return np.all(residual == 0)


def worker_collect_solutions(args):
    """
    单个 worker：运行 G6K pump，收集所有 ℓ∞ ≤ 16 的解（不管 ℓ₂）。
    返回找到的所有解列表。
    """
    worker_id, seed, A, beta_list, gamma_threshold = args

    try:
        from g6k import Siever, SieverParams
        from g6k.algorithms.pump import pump
        from g6k.utils.stats import SieveTreeTracer
    except ImportError as e:
        print(f"[W{worker_id}] G6K import error: {e}")
        return []

    solutions = []

    B_fp = build_sis_basis_fpylll(A)
    LLL.reduction(B_fp)
    BKZ.reduction(B_fp, BKZ.Param(block_size=95, max_loops=4,
                                   flags=BKZ.AUTO_ABORT))

    M = GSO.Mat(B_fp, float_type='double')
    M.update_gso()

    from g6k import SieverParams
    params = SieverParams(reserved_n=DIM, otf_lift=True, threads=8, seed=seed)
    g6k = Siever(M, params)

    def d4f(beta): return max(0, int(11.5 + 0.075 * beta))

    for beta in beta_list:
        for kappa in range(0, 6):
            try:
                tracer = SieveTreeTracer(g6k, root_eps=0.0, start_clocks=False)
                pump(g6k, tracer, kappa, beta, d4f(beta))
            except Exception:
                pass

            # 检查当前格基行
            M.update_gso()
            for i in range(DIM):
                row = np.array([B_fp[i][j] for j in range(DIM)], dtype=np.int64)
                linf = int(np.max(np.abs(row)))
                if linf > gamma_threshold:
                    continue
                l2sq = int(np.sum(row ** 2))
                if l2sq == 0:
                    continue
                v, u = row[:N], row[N:]
                residual = (A @ v + u) % Q
                if np.all(residual == 0):
                    solutions.append(row.copy())
                    print(f"[W{worker_id}] β={beta} κ={kappa}: ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.2f}, ℓ₂≥120:{'✓' if l2sq>=TARGET_L2_SQ else '✗'}")
                    if l2sq >= TARGET_L2_SQ:
                        return solutions  # 直接找到满足全部条件的解

    return solutions


def combine_solutions(solutions, A):
    """
    尝试两解组合：若 x₀ + x₁ 满足 ℓ∞ ≤ 16 AND ℓ₂ ≥ 120，则输出。
    对于每对 (x₀, x₁)，检查 x₀ + x₁ 是否满足条件。
    注意：x₀ + x₁ 也是有效 SIS 解（格点加法）。
    """
    print(f"\n[Combine] Trying {len(solutions)}² = {len(solutions)**2} pairs...")
    sols = np.array(solutions, dtype=np.int64)

    best_l2sq = 0
    best_pair = None

    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            combined = sols[i] + sols[j]
            linf = int(np.max(np.abs(combined)))
            if linf > GAMMA:
                continue
            l2sq = int(np.sum(combined ** 2))
            v, u = combined[:N], combined[N:]
            residual = (A @ v + u) % Q
            if np.all(residual == 0):
                if l2sq >= TARGET_L2_SQ:
                    print(f"[Combine] ✅ Found! pair ({i},{j}): ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.2f}")
                    return combined
                if l2sq > best_l2sq:
                    best_l2sq = l2sq
                    best_pair = (i, j, combined.copy())

    if best_pair:
        i, j, c = best_pair
        print(f"[Combine] Best pair ({i},{j}): ℓ₂={np.sqrt(best_l2sq):.2f} (still < 120)")

    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem5.txt'
    A = load_problem(path)
    print(f"Loaded A: {A.shape}, q={Q}, γ={GAMMA}")

    # Phase 1: 收集大量解（包括 ℓ∞ ≤ 16 的全部解）
    # 扩展 beta 范围，让筛法更彻底地探索
    beta_list = list(range(90, 130, 5))
    print(f"Beta schedule: {beta_list}")

    NUM_WORKERS = 40
    tasks = [
        (wid, wid * 99991 + 7, A, beta_list, GAMMA)
        for wid in range(NUM_WORKERS)
    ]

    print(f"\nPhase 1: Collecting solutions with ℓ∞ ≤ {GAMMA}...")
    print(f"(Any solution with ℓ₂ ≥ 120 will terminate immediately)")
    t0 = time.time()

    all_solutions = []
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_collect_solutions, tasks)

    for r in results:
        for sol in r:
            l2sq = int(np.sum(sol.astype(np.int64) ** 2))
            if l2sq >= TARGET_L2_SQ and verify_linf_only(sol, A):
                print(f"\n✅ PHASE 1 SUCCESS! ℓ∞={np.max(np.abs(sol))}, ℓ₂={np.sqrt(l2sq):.2f}")
                np.save('p5_solution.npy', sol)
                print(f"Saved. Total time: {time.time()-t0:.0f}s")
                return sol
            if verify_linf_only(sol, A):
                all_solutions.append(sol)

    print(f"\nPhase 1 done in {time.time()-t0:.0f}s")
    print(f"Collected {len(all_solutions)} solutions with ℓ∞ ≤ {GAMMA}")

    if not all_solutions:
        print("No solutions collected. Try running again.")
        return None

    # 打印 ℓ₂ 分布
    l2_values = sorted([np.sqrt(int(np.sum(s**2))) for s in all_solutions])
    print(f"ℓ₂ range: [{l2_values[0]:.2f}, {l2_values[-1]:.2f}]")
    print(f"ℓ₂ distribution: {[(round(v,1)) for v in l2_values[:10]]}...")

    # Phase 2: 尝试两解组合
    print(f"\nPhase 2: Trying pairwise combinations...")
    result = combine_solutions(all_solutions, A)
    if result is not None:
        l2sq = int(np.sum(result.astype(np.int64) ** 2))
        print(f"\n✅ PHASE 2 SUCCESS! ℓ∞={np.max(np.abs(result))}, ℓ₂={np.sqrt(l2sq):.2f}")
        np.save('p5_solution.npy', result)
        print(f"Saved. Total time: {time.time()-t0:.0f}s")
        return result

    # Phase 3: 差值组合（x₀ - x₁）
    print(f"\nPhase 3: Trying difference combinations...")
    sols = np.array(all_solutions, dtype=np.int64)
    for i in range(len(all_solutions)):
        for j in range(len(all_solutions)):
            if i == j:
                continue
            diff = sols[i] - sols[j]
            linf = int(np.max(np.abs(diff)))
            if linf > GAMMA:
                continue
            l2sq = int(np.sum(diff.astype(np.int64) ** 2))
            if l2sq == 0:
                continue
            v, u = diff[:N], diff[N:]
            residual = (A @ v + u) % Q
            if np.all(residual == 0) and l2sq >= TARGET_L2_SQ:
                print(f"✅ PHASE 3 SUCCESS! diff ({i},{j}): ℓ∞={linf}, ℓ₂={np.sqrt(l2sq):.2f}")
                np.save('p5_solution.npy', diff)
                return diff

    print("❌ All phases failed. Consider running more workers or higher beta.")
    return None


if __name__ == '__main__':
    main()
