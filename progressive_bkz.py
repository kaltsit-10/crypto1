"""
渐进BKZ求解器：SIS Problem 1 (q=100, gamma=15, dim=200)

依赖: numpy, fpylll, default.json (BKZ策略文件)
用法: python3 progressive_bkz.py

从原始格基出发，依次运行 LLL → BKZ-40 → 50 → 55 → 60 → 65 → 70 → 75 → 80。
每阶段保存中间基，找到解自动停止并写入 solution_p1.txt。

实测时间（单机，fpylll枚举法）:
  BKZ-40: ~3s     BKZ-50: ~15s     BKZ-55: ~30s
  BKZ-60: ~50s    BKZ-65: ~105s    BKZ-70: ~7min
  BKZ-75: ~25min  BKZ-80: ~1-2h (估计)
  总计: ~2-3h (从零开始到BKZ-80)
"""
import numpy as np
import time, os
from fpylll import IntegerMatrix, LLL, BKZ

# === 配置（队友可修改路径） ===
BASE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
A = np.load(os.path.join(BASE, 'problem1_A.npy'))
STRATEGIES = os.path.join(BASE, 'default.json')

q, m, n, gamma, dim = 100, 100, 100, 15, 200
BLOCK_SIZES = [40, 50, 55, 60, 65, 70, 75, 80]

t_start = time.time()

def elapsed():
    return time.time() - t_start

def centered_u(u_raw):
    u = u_raw % q
    return np.where(u >= q//2, u - q, u).astype(np.int64)

def report_best(B_mat, tag=""):
    best_s, best_lv, best_lu = 999, 0, 0
    for i in range(dim):
        v = np.array([int(B_mat[i, j]) for j in range(m)], dtype=np.int64)
        u_raw = np.array([int(B_mat[i, j]) for j in range(m, dim)], dtype=np.int64)
        if np.all(v == 0):
            continue
        u = centered_u(u_raw)
        lv = int(np.max(np.abs(v)))
        lu = int(np.max(np.abs(u)))
        s = max(lv, lu)
        if s < best_s:
            best_s, best_lv, best_lu = s, lv, lu
    print(f"  [{tag}] score={best_s} lv={best_lv} lu={best_lu} (t={elapsed():.1f}s)", flush=True)
    return best_s, best_lv, best_lu

def save_solution(B_mat):
    """Scan basis for rows satisfying linf ≤ gamma, save if found."""
    for i in range(dim):
        v = np.array([int(B_mat[i, j]) for j in range(m)], dtype=np.int64)
        u_raw = np.array([int(B_mat[i, j]) for j in range(m, dim)], dtype=np.int64)
        if np.all(v == 0):
            continue
        u = centered_u(u_raw)
        lv = int(np.max(np.abs(v)))
        lu = int(np.max(np.abs(u)))
        if lv <= gamma and lu <= gamma and (lv > 0 or lu > 0):
            valid = np.all((A @ v + u_raw) % q == 0)
            if valid:
                l2v = float(np.linalg.norm(v.astype(float)))
                l2u = float(np.linalg.norm(u.astype(float)))
                out = f"v = {v.tolist()}\nu = {u.tolist()}\n"
                out += f"# ||v||_inf = {lv}\n# ||u||_inf = {lu}\n"
                out += f"# ||v||_2^2 + ||u||_2^2 = {l2v**2 + l2u**2:.0f}\n"
                path = os.path.join(BASE, 'solution_p1.txt')
                with open(path, 'w') as f:
                    f.write(out)
                print(f"  *** SOLUTION SAVED to {path}! ***", flush=True)
                return True
    return False

print("=" * 60)
print("Progressive BKZ Solver — SIS Problem 1")
print(f"q={q}, m={m}, n={n}, dim={dim}, gamma={gamma}")
print(f"Block sizes: {BLOCK_SIZES}")
print(f"Strategies: {STRATEGIES}")
print("=" * 60)

# === Phase 0: Build lattice basis ===
print("\n--- Phase 0: Building lattice basis ---", flush=True)
basis = np.zeros((dim, dim), dtype=np.int64)
for i in range(m):
    basis[i, i] = 1
    for j in range(n):
        basis[i, m + j] = -int(A[j, i])
for j in range(n):
    basis[m + j, m + j] = q

valid_rows = sum(1 for i in range(dim) if np.all((A @ basis[i, :m] + basis[i, m:]) % q == 0))
print(f"  Valid rows: {valid_rows}/{dim}", flush=True)

M = IntegerMatrix(dim, dim)
for i in range(dim):
    for j in range(dim):
        M[i, j] = int(basis[i, j])

report_best(M, "initial")

# === Phase 1: LLL ===
print("\n--- Phase 1: LLL (delta=0.99) ---", flush=True)
LLL.reduction(M, delta=0.99)
report_best(M, "LLL")

# === Phase 2: Progressive BKZ ===
for bs in BLOCK_SIZES:
    print(f"\n--- Phase 2: BKZ-{bs} ---", flush=True)
    t1 = time.time()
    try:
        BKZ.reduction(M, BKZ.Param(
            block_size=bs,
            strategies=STRATEGIES,
            max_loops=2 if bs >= 55 else 1,
            flags=BKZ.DEFAULT | BKZ.GH_BND if bs >= 55 else BKZ.DEFAULT
        ))
        dt = time.time() - t1
        s, lv, lu = report_best(M, f"BKZ-{bs}")
        print(f"  BKZ-{bs} took {dt:.1f}s", flush=True)

        # Save intermediate basis
        B_save = np.zeros((dim, dim), dtype=np.int64)
        for i in range(dim):
            for j in range(dim):
                B_save[i, j] = int(M[i, j])
        path = os.path.join(BASE, f'bkz{bs}_sis_basis.npy')
        np.save(path, B_save)
        print(f"  Saved {path}", flush=True)

        # Early exit if solution found in basis
        if save_solution(M):
            break

    except Exception as e:
        print(f"  BKZ-{bs} CRASHED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print(f"  Continuing with last good basis...", flush=True)
        break

# === Final Report ===
elapsed_total = elapsed()
print(f"\n{'='*60}")
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
final_s, final_lv, final_lu = report_best(M, "FINAL")
print(f"gamma={gamma}, gap={final_s - gamma}")

if final_s <= gamma:
    print("*** SOLUTION FOUND ***")
else:
    print(f"Best score={final_s}, need {final_s - gamma} more points. "
          f"Try higher BKZ blocks or combinatorial search.")
