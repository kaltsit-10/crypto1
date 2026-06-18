"""
通用验证脚本：验证第五题的解是否满足全部条件
"""

import numpy as np
import ast
import sys

Q = 120
N = M = Q
GAMMA = 16
TARGET_L2_SQ = Q * Q  # 14400


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    return A_cols.T  # (n, m)


def load_solution(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.int64)
    else:
        with open(path) as f:
            content = f.read().strip()
        return np.array(ast.literal_eval(content), dtype=np.int64)


def verify_p5(sol, A, verbose=True):
    if len(sol) != 2 * N:
        print(f"❌ Wrong dimension: {len(sol)} (expected {2*N})")
        return False

    v, u = sol[:N], sol[N:]

    # 条件1: SIS 方程
    residual = (A @ v + u) % Q
    sis_ok = np.all(residual == 0)

    # 条件2: 无穷范数
    linf_v = int(np.max(np.abs(v)))
    linf_u = int(np.max(np.abs(u)))
    linf = max(linf_v, linf_u)
    linf_ok = linf <= GAMMA

    # 条件3: 欧氏范数下界
    l2sq = int(np.sum(sol ** 2))
    l2 = float(np.sqrt(l2sq))
    l2_ok = l2sq >= TARGET_L2_SQ

    if verbose:
        print("=== P5 Solution Verification ===")
        print(f"  SIS equation Av + u ≡ 0 (mod {Q}): {'✅' if sis_ok else '❌'} (max residual: {np.max(np.abs(residual))})")
        print(f"  ℓ∞(v) = {linf_v} ≤ {GAMMA}: {'✅' if linf_v <= GAMMA else '❌'}")
        print(f"  ℓ∞(u) = {linf_u} ≤ {GAMMA}: {'✅' if linf_u <= GAMMA else '❌'}")
        print(f"  ℓ₂(u,v) = {l2:.4f} ≥ {Q}: {'✅' if l2_ok else '❌ (gap: ' + f'{Q - l2:.2f})'}")
        print(f"  ℓ₂² = {l2sq} ≥ {TARGET_L2_SQ}: {'✅' if l2_ok else '❌'}")

        if not (linf_ok and l2_ok):
            # 计算部分分
            gamma_prime = linf
            if linf_ok and not l2_ok:
                score = f"满分 10/10（ℓ∞条件满足，但ℓ₂条件不满足，题目未说明扣分方式）"
            elif not linf_ok:
                partial = max(0, 10 - 2 * gamma_prime + 2 * GAMMA)
                score = f"部分分 {partial}/10 (γ'={gamma_prime})"
            else:
                score = "满分 10/10"
            print(f"  得分估计: {score}")

        all_ok = sis_ok and linf_ok and l2_ok
        print(f"\n  总体: {'✅ 完全满足 P5 条件（满分）' if all_ok else '❌ 不满足全部条件'}")

        # 统计信息
        abs_sol = np.abs(sol)
        print(f"\n  解向量统计:")
        print(f"    非零元素: {np.sum(abs_sol > 0)}/240")
        print(f"    平均|x_i|: {np.mean(abs_sol):.2f}")
        for t in [0, 4, 8, 12, 16]:
            cnt = np.sum(abs_sol <= t)
            print(f"    |x_i| ≤ {t:2d}: {cnt:3d} / 240")

    return sis_ok and linf_ok and l2_ok


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify.py <solution_file> [problem_file]")
        print("  solution_file: .npy or text file with solution vector")
        print("  problem_file: problem5.txt (default)")
        sys.exit(1)

    sol_path = sys.argv[1]
    prob_path = sys.argv[2] if len(sys.argv) > 2 else '../sis_inf_problems/problem5.txt'

    A = load_problem(prob_path)
    sol = load_solution(sol_path)

    print(f"Problem: {prob_path}")
    print(f"Solution: {sol_path} (dim={len(sol)})")
    print()

    verify_p5(sol, A)
