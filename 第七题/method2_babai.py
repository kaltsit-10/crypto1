"""
方案二：BKZ 约化 + Babai 最近平面 CVP（最快，推荐首选）

思路：
  非齐次 SIS 等价于 CVP 问题：在 SIS 格 L₀ 中找最近格点到目标 w₀=(0,t)。
  BKZ 约化格基后，Babai 最近平面（Nearest Plane）算法近似求解 CVP。
  对 dim=280、GH≈47.9、γ=17 的问题：BKZ-100+ 后 Babai 的误差 ≈ GH，
  P(ℓ∞ ≤ 17 | ℓ₂ ≈ GH) ≈ 1，因此高概率直接给出满足条件的解。

预期时间：10-60 分钟
硬件需求：纯 fpylll，不需要 G6K
"""

import numpy as np
import ast
import time
import sys
from fpylll import IntegerMatrix, GSO, LLL, BKZ

Q = 140
N = M = Q
GAMMA = 17
DIM = N + M  # 280


def load_problem(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    A_str = lines[0][lines[0].index('['):]
    t_str = lines[1][lines[1].index('['):]
    A_cols = np.array(ast.literal_eval(A_str), dtype=np.int64)
    t = np.array(ast.literal_eval(t_str), dtype=np.int64)
    return A_cols.T, t  # A: (n, m), t: (n,)


def build_sis_basis_fp(A):
    """构造 280×280 SIS 齐次格基（fpylll IntegerMatrix）"""
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


def center_lift(v, q):
    """将向量模 q 中心化到 [-q/2, q/2]"""
    v = np.array(v, dtype=np.int64) % q
    v = np.where(v > q // 2, v - q, v)
    return v


def verify(v, u, A, t):
    v = np.array(v, dtype=np.int64)
    u = np.array(u, dtype=np.int64)
    residual = (A @ v + u - t) % Q
    linf_v = int(np.max(np.abs(v)))
    linf_u = int(np.max(np.abs(u)))
    linf = max(linf_v, linf_u)
    sis_ok = np.all(residual == 0)
    print(f"  SIS: {'✅' if sis_ok else '❌'}, ℓ∞(v)={linf_v}, ℓ∞(u)={linf_u}, ℓ∞={linf}≤{GAMMA}:{'✅' if linf<=GAMMA else '❌'}")
    return sis_ok and linf <= GAMMA


def babai_nearest_plane(M, target):
    """
    Babai 最近平面算法（手动实现，适用于 fpylll GSO 对象）
    target: 目标向量（整数列表，长度 dim）
    返回：最近格点坐标（lattice basis 的整数系数向量）
    """
    n = M.d
    v = [float(x) for x in target]

    # 从最后一个基向量开始，逐步投影
    coeffs = [0] * n
    for i in range(n - 1, -1, -1):
        # 计算 v 在 b_i* 上的分量
        proj = sum(v[j] * M.get_mu(j, i) if j > i else (v[j] * 1.0 if j == i else 0.0)
                   for j in range(n))
        # 实际上用 GSO 的 get_r 和 mu
        # 计算 <v, b_i*> / <b_i*, b_i*>
        bi_star_sq = M.get_r(i, i)
        if bi_star_sq == 0:
            c_i = 0
        else:
            # 内积 <v, b_i*>
            dot = 0.0
            for j in range(n):
                dot += float(v[j]) * M.get_mu(j, i) if j > i else float(v[i]) if j == i else 0.0
            # Actually need to compute properly using GS coordinates
            # Use M.babai coefficient
            c_i = round(M.get_r(i, i))  # placeholder

        coeffs[i] = 0  # will be computed below

    return coeffs


def babai_cvp_fpylll(B_fp, target_np):
    """
    使用 fpylll 的 CVP 求解器
    B_fp: fpylll IntegerMatrix (reduced basis)
    target_np: target vector as numpy int64 array
    """
    from fpylll import CVP
    target_list = [int(x) for x in target_np]
    # fpylll CVP
    try:
        result = CVP.closest_vector(B_fp, target_list)
        return np.array(result, dtype=np.int64)
    except Exception as e:
        print(f"CVP failed: {e}")
        return None


def manual_babai(B_reduced_np, target_np):
    """
    手动实现 Babai 最近平面算法（纯 numpy）
    B_reduced_np: 约化格基，行向量，shape (dim, dim)
    target_np: 目标向量，shape (dim,)
    返回最近格点（与 target 最近的格中点）
    """
    dim = B_reduced_np.shape[0]
    B = B_reduced_np.astype(np.float64)
    t = target_np.astype(np.float64)

    # Gram-Schmidt 正交化
    B_star = np.zeros_like(B)
    mu = np.zeros((dim, dim))
    for i in range(dim):
        B_star[i] = B[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
            B_star[i] -= mu[i, j] * B_star[j]

    # Babai 最近平面
    v = t.copy()
    coeffs = np.zeros(dim, dtype=np.int64)
    for i in range(dim - 1, -1, -1):
        bstar_norm_sq = np.dot(B_star[i], B_star[i])
        if bstar_norm_sq < 1e-10:
            continue
        c = np.dot(v, B_star[i]) / bstar_norm_sq
        c_round = int(round(c))
        coeffs[i] = c_round
        v -= c_round * B[i]

    # 重构最近格点
    closest = np.dot(coeffs, B_reduced_np)
    return closest.astype(np.int64)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../sis_inf_problems/problem7.txt'
    A, t = load_problem(path)
    print(f"Loaded: A {A.shape}, t {t.shape}, t range [{t.min()}, {t.max()}]")
    print(f"q={Q}, γ={GAMMA}")

    # 目标向量：(0^n, t^n) 即 v=0, u=t 对应的格嵌入
    # 在格的坐标中：目标是 (0,...,0, t[0],...,t[n-1])
    target = np.zeros(DIM, dtype=np.int64)
    target[N:] = t  # 后半段是 t

    # 中心化 t 到 [-q/2, q/2]
    target_centered = center_lift(target, Q)
    print(f"Target ℓ₂ = {np.linalg.norm(target_centered):.2f}")

    # 构造 SIS 格基
    print("\nBuilding SIS basis (280×280)...")
    B_fp = build_sis_basis_fp(A)

    # LLL
    print("Running LLL...")
    t0 = time.time()
    LLL.reduction(B_fp)
    print(f"LLL done in {time.time()-t0:.1f}s")

    # 逐步提高 BKZ，每次尝试 Babai CVP
    for beta in [60, 70, 80, 90, 95, 100, 105, 110]:
        print(f"\nRunning BKZ-{beta}...")
        t0 = time.time()
        BKZ.reduction(B_fp, BKZ.Param(
            block_size=beta,
            max_loops=4,
            flags=BKZ.AUTO_ABORT | BKZ.VERBOSE
        ))
        elapsed = time.time() - t0
        print(f"BKZ-{beta} done in {elapsed:.1f}s")

        # 获取约化基
        B_np = np.array([[B_fp[i][j] for j in range(DIM)] for i in range(DIM)], dtype=np.int64)

        # 检查前几行是否直接满足（有时 BKZ 会输出接近目标的行）
        # 尝试1：fpylll CVP
        print(f"Trying fpylll CVP...")
        closest = babai_cvp_fpylll(B_fp, target_centered)
        if closest is not None:
            diff = target_centered - closest  # = (0,t) - lattice_point = (-v, u-t)? 需要仔细
            # diff = target - B*coeffs
            # 如果格点 x 满足 Ax_v + x_u ≡ 0 (mod q)，且 target - x = (v', u')
            # 则 A·(-v') + (-u') ≡ 0 (mod q)，即 A·v' + u' ≡ 0 but also
            # target = (0, t), x = (x_v, x_u), diff = (-x_v, t - x_u)
            # Let v = -x_v, u_error = t - x_u
            # A*v + u_error = A*(-x_v) + t - x_u = t - (A*x_v + x_u) ≡ t (mod q) if A*x_v + x_u ≡ 0

            # So if x is in L_0: A*(-diff_v) + (t - diff_u) ≡ t (mod q) ✓
            # Wait: diff = target - closest = (0 - x_v, t - x_u) = (-x_v, t - x_u)
            # v = -diff[:N] = x_v, u = t - diff[N:] ... hmm, let's be careful

            # closest is a vector in L_0: A*(closest_v) + closest_u ≡ 0 (mod q)
            # diff = target_centered - closest = (0 - closest_v, t_c - closest_u) = (-closest_v, t_c - closest_u)
            # Set v = -closest_v, u_shift = t_c - closest_u
            # Check: A*v + u_shift = A*(-closest_v) + t_c - closest_u
            #       = t_c - (A*closest_v + closest_u) ≡ t_c (mod q)
            # But we need A*v + u ≡ t (mod q), and t_c = t - q*round(t/q) (centered)
            # So we need to be careful about the center lifting

            # Simpler: just use diff directly as a SIS vector and check
            v_cand = -closest[:N]
            u_cand = center_lift(t - closest[N:], Q)

            print(f"  CVP candidate: ℓ∞(v)={np.max(np.abs(v_cand))}, ℓ∞(u)={np.max(np.abs(u_cand))}")
            if verify(v_cand, u_cand, A, t):
                print(f"\n✅ SOLUTION FOUND at BKZ-{beta}!")
                sol = np.concatenate([v_cand, u_cand])
                np.save('p7_solution.npy', sol)
                print(f"Saved to p7_solution.npy")
                return sol

        # 尝试2：手动 Babai
        print(f"Trying manual Babai nearest plane...")
        try:
            closest2 = manual_babai(B_np, target_centered)
            v_cand2 = -closest2[:N]
            u_cand2 = center_lift(t - closest2[N:], Q)
            print(f"  Babai candidate: ℓ∞(v)={np.max(np.abs(v_cand2))}, ℓ∞(u)={np.max(np.abs(u_cand2))}")
            if verify(v_cand2, u_cand2, A, t):
                print(f"\n✅ SOLUTION FOUND via Babai at BKZ-{beta}!")
                sol = np.concatenate([v_cand2, u_cand2])
                np.save('p7_solution.npy', sol)
                return sol
        except Exception as e:
            print(f"  Babai failed: {e}")

        # 保存当前约化基供后续使用
        np.save(f'/tmp/p7_bkz{beta}_basis.npy', B_np)
        print(f"  Saved basis to /tmp/p7_bkz{beta}_basis.npy")

    print("\n❌ BKZ+Babai failed. Try method1_kannan.py (G6K pump).")
    return None


if __name__ == '__main__':
    main()
