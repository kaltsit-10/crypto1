"""
通用验证脚本：验证第七题的解是否满足全部条件
"""

import numpy as np
import ast
import sys

Q = 140
N = M = Q
GAMMA = 17


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    t_str = lines[1][lines[1].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    t_vec = np.array(ast.literal_eval(t_str), dtype=np.int64)
    return A_cols.T, t_vec


def load_solution(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.int64)
    with open(path) as f:
        content = f.read().strip()
    return np.array(ast.literal_eval(content), dtype=np.int64)


def verify_p7(sol, A, t, verbose=True):
    if len(sol) != 2 * N:
        print(f"❌ Wrong dimension: {len(sol)} (expected {2*N})")
        return False

    v, u = sol[:N], sol[N:]

    # 条件1: 非齐次 SIS 方程 Av + u ≡ t (mod q)
    residual = (A @ v + u - t) % Q
    sis_ok = np.all(residual == 0)

    # 条件2: 无穷范数 ≤ γ
    linf_v = int(np.max(np.abs(v)))
    linf_u = int(np.max(np.abs(u)))
    linf = max(linf_v, linf_u)
    linf_ok = linf <= GAMMA

    if verbose:
        print("=== P7 Solution Verification ===")
        print(f"  Non-homogeneous SIS: Av + u ≡ t (mod {Q}): {'✅' if sis_ok else '❌'}")
        if not sis_ok:
            print(f"    Max residual = {np.max(np.abs(residual))}")
        print(f"  ℓ∞(v) = {linf_v} ≤ {GAMMA}: {'✅' if linf_v <= GAMMA else '❌'}")
        print(f"  ℓ∞(u) = {linf_u} ≤ {GAMMA}: {'✅' if linf_u <= GAMMA else '❌'}")

        l2 = float(np.sqrt(np.sum(sol.astype(np.float64)**2)))
        print(f"  ℓ₂(u,v) = {l2:.4f}")

        all_ok = sis_ok and linf_ok
        if all_ok:
            print(f"\n  ✅ 完全满足 P7 条件（满分 10/10）")
        else:
            gamma_prime = linf
            partial = max(0, 10 - 2 * gamma_prime + 2 * GAMMA)
            print(f"\n  ❌ 不满足 ℓ∞ ≤ {GAMMA}: γ' = {gamma_prime}")
            print(f"  部分分: max{{0, 10 - 2×{gamma_prime} + 2×{GAMMA}}} = {partial}/10")

        # 统计
        abs_sol = np.abs(sol)
        print(f"\n  解向量统计（dim=280）:")
        print(f"    非零元素: {np.sum(abs_sol > 0)}/280")
        print(f"    平均|x_i|: {np.mean(abs_sol):.2f}")
        for threshold in [0, 4, 8, 12, 17]:
            cnt = np.sum(abs_sol <= threshold)
            print(f"    |x_i| ≤ {threshold:2d}: {cnt:3d} / 280")

    return sis_ok and linf_ok


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify.py <solution_file> [problem_file]")
        print("  solution_file: .npy or text file with 280-dim solution (v||u)")
        sys.exit(1)

    sol_path = sys.argv[1]
    prob_path = sys.argv[2] if len(sys.argv) > 2 else '../sis_inf_problems/problem7.txt'

    A, t = load_problem(prob_path)
    sol = load_solution(sol_path)

    print(f"Problem: {prob_path}")
    print(f"Solution: {sol_path} (dim={len(sol)})")
    print()
    verify_p7(sol, A, t)
