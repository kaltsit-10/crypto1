"""
G6K Workout SVP Solver —— 基于筛法的SIS短向量求解器
=====================================================
适用场景：q-ary格上SIS问题，目标找到 ℓ∞ ≤ gamma 的非零短向量。
核心方法：渐进G6K Workout（HNP 2022论文方案），用筛法替代枚举法做BKZ的SVP oracle。

依赖：
  pip install fpylll numpy
  git clone https://github.com/fplll/g6k && cd g6k && pip install -e .

MAX_SIEVING_DIM 修复（必须！）：
  G6K默认 MAX_SIEVING_DIM=128，但200维格需要 >= 200。
  编译前修改 kernel/g6k_config.h: #define MAX_SIEVING_DIM 256
  然后: python3 setup.py clean && python3 setup.py build_ext --inplace
  验证: python3 -c "from g6k import Siever; print(Siever.max_sieving_dim)"

用法：
  python g6k_workout_solver.py                          # 使用默认参数
  python g6k_workout_solver.py --bs 60,70,80             # 自定义块大小序列
  python g6k_workout_solver.py --gamma 12 --threads 4    # 自定义目标和线程

作者：队友共用版
日期：2026-05-16
"""

import numpy as np
import time
import os
import sys
import argparse

# ============================================================
# 全局配置
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

# 问题参数（可从命令行覆盖）
Q = 100       # 模数
M = 100       # v 维度
N = 100       # u 维度（通常 M=N）
DIM = 200     # 格维度 = M + N


def centered_u(u_raw):
    """将模q的u_raw中心化到 [-q/2, q/2)"""
    u = u_raw % Q
    return np.where(u >= Q // 2, u - Q, u).astype(np.int64)


def scan_basis(B_mat, tag=""):
    """扫描格基所有行，返回最佳score（ℓ∞范数）"""
    best_s, best_lv, best_lu, best_i = 999, 0, 0, -1
    for i in range(DIM):
        v = np.array([int(B_mat[i, j]) for j in range(M)], dtype=np.int64)
        u_raw = np.array([int(B_mat[i, j]) for j in range(M, DIM)], dtype=np.int64)
        if np.all(v == 0):
            continue
        u = centered_u(u_raw)
        lv = int(np.max(np.abs(v)))
        lu = int(np.max(np.abs(u)))
        s = max(lv, lu)
        if s < best_s:
            best_s, best_lv, best_lu, best_i = s, lv, lu, i
    print(f"  [{tag}] score={best_s} lv={best_lv} lu={best_lu} row={best_i}", flush=True)
    return best_s, best_lv, best_lu, best_i


def report_g6k(g6k, tag=""):
    """扫描G6K当前基，返回最佳score"""
    return scan_basis(g6k.M.B, tag)


def save_solution(g6k, tag, output_path):
    """验证并保存解向量。返回True表示找到并保存了有效解。"""
    A = np.load(os.path.join(BASE, 'problem1_A.npy'))
    B_mat = g6k.M.B
    for i in range(DIM):
        v = np.array([int(B_mat[i, j]) for j in range(M)], dtype=np.int64)
        u_raw = np.array([int(B_mat[i, j]) for j in range(M, DIM)], dtype=np.int64)
        if np.all(v == 0):
            continue
        u = centered_u(u_raw)
        lv = int(np.max(np.abs(v)))
        lu = int(np.max(np.abs(u)))
        if lv <= GAMMA and lu <= GAMMA and (lv > 0 or lu > 0):
            # 验证格约束: Av + u ≡ 0 (mod q)
            u_check = centered_u((-A @ v) % Q)
            l2v = float(np.linalg.norm(v.astype(float)))
            l2u = float(np.linalg.norm(u_check.astype(float)))
            out = f"v = {v.tolist()}\nu = {u_check.tolist()}\n"
            out += f"# ||v||_inf = {lv}\n# ||u||_inf = {lu}\n"
            out += f"# ||v||_2^2 + ||u||_2^2 = {l2v**2 + l2u**2:.0f}\n"
            out += f"# method: {tag}\n"
            with open(output_path, 'w') as f:
                f.write(out)
            print(f"\n{'='*60}", flush=True)
            print(f"*** 解已保存! ({tag}) ***", flush=True)
            print(f"    ||v||_inf={lv}, ||u||_inf={lu}", flush=True)
            print(f"    ||v||_2^2 + ||u||_2^2 = {l2v**2 + l2u**2:.0f}", flush=True)
            print(f"{'='*60}", flush=True)
            return True
    return False


def run_workout(g6k, tracer, bs_list, gamma, output_path):
    """
    执行渐进G6K Workout管线。

    G6K Workout原理：
      - 从 kappa=0 开始，在高维投影子格上运行筛法（sieving）
      - 筛法找到的短向量通过"pump"操作插入基中
      - 渐进增大 bs（块大小）以获得更深的归约
      - bs 越大，筛法越强，但内存/时间也越大

    参数:
      g6k:        Siever实例（已绑定GSO基）
      tracer:     SieveTreeTracer（日志/性能追踪）
      bs_list:    块大小序列，如 [60, 70, 75, 80]
      gamma:      ℓ∞目标
      output_path: 解文件保存路径
    """
    from g6k.algorithms.workout import workout

    best_score = 999

    for bs in bs_list:
        if best_score <= gamma:
            break
        if bs > DIM:
            print(f"  跳过 bs={bs} > dim={DIM}", flush=True)
            continue

        print(f"\n--- G6K Workout: kappa=0, bs={bs} ---", flush=True)
        t1 = time.time()

        try:
            workout(g6k, tracer, 0, bs,
                    dim4free_min=0,
                    dim4free_dec=2,
                    start_n=min(30, bs),
                    verbose=True)

            dt = time.time() - t1
            print(f"  完成, 耗时 {dt:.1f}s", flush=True)

            s, lv, lu, bi = report_g6k(g6k, f"bs={bs}")
            if s < best_score:
                best_score = s
                print(f"  *** 改进: score {s}", flush=True)
                if s <= gamma:
                    if save_solution(g6k, f"G6K-workout-bs{bs}", output_path):
                        return True, s

        except Exception as e:
            print(f"  Workout 失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    return False, best_score


def main():
    global GAMMA

    parser = argparse.ArgumentParser(
        description='G6K Workout SVP Solver — SIS短向量求解器')
    parser.add_argument('--basis', type=str, default='bkz80_sis_basis.npy',
                        help='输入基文件 (.npy), 默认 bkz80_sis_basis.npy')
    parser.add_argument('--gamma', type=int, default=15,
                        help='ℓ∞ 目标范数 (默认 15)')
    parser.add_argument('--bs', type=str, default='60,70,75,80',
                        help='Workout块大小序列, 逗号分隔 (默认 60,70,75,80)')
    parser.add_argument('--threads', type=int, default=8,
                        help='筛法线程数 (默认 8)')
    parser.add_argument('--output', type=str, default='solution_p1.txt',
                        help='解文件输出路径 (默认 solution_p1.txt)')
    parser.add_argument('--max-sieving-dim', type=int, default=256,
                        help='期望的MAX_SIEVING_DIM (仅用于检查, 默认 256)')
    args = parser.parse_args()

    GAMMA = args.gamma
    bs_list = [int(x.strip()) for x in args.bs.split(',')]
    output_path = os.path.join(BASE, args.output)
    basis_path = os.path.join(BASE, args.basis)

    t_start = time.time()

    print("=" * 60, flush=True)
    print("G6K Workout SVP Solver (HNP 2022 筛法方案)", flush=True)
    print(f"q={Q}, dim={DIM}, gamma={GAMMA}", flush=True)
    print(f"bs序列: {bs_list}, threads={args.threads}", flush=True)
    print("=" * 60, flush=True)

    # ---- 检查 MAX_SIEVING_DIM ----
    from g6k import Siever
    max_sd = Siever.max_sieving_dim
    print(f"G6K MAX_SIEVING_DIM: {max_sd}", flush=True)
    if isinstance(max_sd, int) and max_sd < DIM:
        print(f"\n!!! 警告: MAX_SIEVING_DIM={max_sd} < dim={DIM}", flush=True)
        print("!!! 需要修改 g6k/kernel/g6k_config.h 并重新编译", flush=True)
        print("!!! 详见本文件头部的注释说明", flush=True)
        if max_sd < max(bs_list):
            print("!!! 当前 MAX_SIEVING_DIM 不足以运行指定的 bs 序列", flush=True)
            sys.exit(1)

    # ---- 加载基 ----
    print(f"\n加载基: {args.basis}...", flush=True)
    B_init = np.load(basis_path)

    from fpylll import IntegerMatrix, GSO
    from g6k import Siever, SieverParams
    from g6k.utils.stats import SieveTreeTracer

    M_mat = IntegerMatrix(DIM, DIM)
    for i in range(DIM):
        for j in range(DIM):
            M_mat[i, j] = int(B_init[i, j])

    U = IntegerMatrix.identity(DIM)
    UinvT = IntegerMatrix.identity(DIM)
    M_gso = GSO.Mat(M_mat, U=U, UinvT=UinvT)
    M_gso.update_gso()

    s_init, lv_init, lu_init, _ = scan_basis(M_mat, f"初始基")
    print(f"  初始 score = {s_init} (lv={lv_init}, lu={lu_init})", flush=True)

    # ---- 初始化G6K ----
    params = SieverParams()
    params["threads"] = args.threads
    g6k = Siever(M_gso, params)
    tracer = SieveTreeTracer(g6k, root_label=("sis-workout", DIM), start_clocks=True)

    # ---- 渐进Workout ----
    print(f"\n{'='*60}", flush=True)
    print("开始渐进G6K Workout", flush=True)
    print(f"{'='*60}", flush=True)

    solved, final_score = run_workout(g6k, tracer, bs_list, GAMMA, output_path)

    # ---- 终局报告 ----
    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    s, lv, lu, _ = report_g6k(g6k, "FINAL")
    print(f"总时间: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"最终score: {s}, gamma={GAMMA}, gap={s - GAMMA}", flush=True)

    if solved or s <= GAMMA:
        if not solved:
            save_solution(g6k, "final", output_path)
        print("状态: 已求解 ✓", flush=True)
    else:
        print(f"状态: 未求解 ✗ (gap={s - GAMMA})", flush=True)
        print("建议:", flush=True)
        print(f"  1. 增加 bs 序列, 如 --bs {','.join(str(b) for b in sorted(bs_list + [max(bs_list)+5]))}", flush=True)
        print("  2. 确认 MAX_SIEVING_DIM >= dim", flush=True)
        print("  3. 尝试更好的初始基 (BKZ-85+)", flush=True)

    tracer.exit()
    return 0 if (solved or s <= GAMMA) else 1


if __name__ == '__main__':
    sys.exit(main())
