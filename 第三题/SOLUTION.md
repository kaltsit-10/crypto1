# P3 题解：ℓ∞-SIS 求解器（高 BS G6K Pump 筛法）

## 问题描述

- **类型**: Short Integer Solution (SIS)，ℓ∞ 约束
- **维度**: 240 (v: 120 维, u: 120 维)
- **模数**: q = 120
- **目标**: 找到非零向量 (v, u) 满足 A·v + u ≡ 0 (mod 120)，且 **ℓ∞ ≤ 16**
- **得分**: 满分 10/10

## 硬件配置

| 项目 | 规格 |
|------|------|
| CPU | 4× AMD EPYC 9654 96-Core @ 2.4-3.7 GHz |
| 总核心数 | 384 物理核心 |
| 内存 | 556 GB DDR5 |
| OS | Ubuntu 22.04, Linux 6.5.0-41 |
| Python | 3.10.12 |
| G6K | 0.1.2 (CPU gauss sieve) |
| fpylll | 0.6.4 |

## 最终结果

```
ℓ∞ = 16  (max|v| = 16, max|u| = 16)
SIS 方程验证: ✅ A·v + u ≡ 0 (mod 120)
求解时间: 446 秒 (~7.5 分钟)
```

仅 4 个坐标恰好在边界 16：v[37]=16, v[83]=16, v[84]=-16, u[67]=16，其余 236 个坐标均 ≤ 15。

## 方法概述

### 核心思路：G6K 高 BS Pump 筛法（跳过 ℓ∞ C++ 补丁）

本题是一个 240 维的 SIS 格问题。传统方法（如 fpylll BKZ）在 SIS 格上会因 Babai 死循环而崩溃。G6K（General Sieve Kernel）的 pump 算法通过 GaussSieve 在局部 block 中寻找短向量，天然避免了 Babai 问题。

**关键洞察**: 不需要在 G6K 的 C++ 层做 ℓ∞ 排序补丁（`patch_g6k_linf.py`）。ℓ2 约化足够强时，ℓ∞ 范数会概率性地随之下降。只需将 pump 的 blocksize 推高，ℓ2 范数足够小后，ℓ∞ ≤ 16 自然出现。

### 算法设计

#### 1. G6K 环境准备

- 使用 `pip install g6k` 安装预编译版本（0.1.2）
- 修补 `bkz.py` 和 `pump.py`：将所有 `g6k.lll()` 调用包裹 `try/except`，防止 SIS 格上的 Babai 死循环

#### 2. 聚焦 Pump 策略（Focused Pump）

全 BKZ tour（`pump_n_jump_bkz_tour`）在 240 维格上处理 ~178 个 block，每个 block 的 sieve 在 ~80 维上下文中运行，单次全 tour 需数小时。

**优化**: 最优向量在基的第 4 行（row 4），只需 pump 前几个 block（κ = 0..5），而非全部 178 个。这样每次 β 级别仅需 6 次 pump 调用，速度提升 30 倍。

```
Focused Pump(basis, β, κ_max=5):
  for β in [90, 95, 100, 105, 110, 115, 120, 125, 130]:
    for κ in [0, 1, 2, 3, 4, 5]:
      pump(g6k, tracer, kappa=κ, blocksize=β, dim4free=11.5+0.075β)
      g6k.lll(0, 240)
      check_linf()
```

#### 3. 大规模并行化（384 核利用）

G6K 的 pump 本身是串行的（逐 block），但每次 pump 调用内部的 GaussSieve 支持多线程（OpenMP）。为充分利用 384 核：

- 启动 **40 个独立 worker 进程**（Python multiprocessing）
- 每个 worker 使用 **8 线程** 进行内部 sieve
- 40 × 8 = **320 核用于 sieve**，剩余 64 核处理 LLL/setup/OS
- 实际 CPU 利用率 ~70%（278 核在跑），空闲时间主要在各 worker 的单线程 LLL 阶段

不同 worker 使用不同随机种子（`seed = worker_id * 100000 + β * 1000 + κ * 10`），探索格基约化的不同随机路径。总并行 pump 上下文数量：40 workers × 9 β 级别 × 6 blocks ≈ **2160 次独立 pump 调用**。

### 求解过程

| 阶段 | β | 时间 | 结果 |
|------|-----|------|------|
| 起点 | 0 | 0s | BKZ-95 基，ℓ∞=18 |
| 突破 #1 | 95 | 78s | W7 找到 ℓ∞=17 |
| 突破 #2 | 100 | ~120s | 多个 worker 稳定 ℓ∞=17 |
| **解决** | **105** | **446s** | **W37 找到 ℓ∞=16** ✅ |

### 为什么能成功

1. **足够高的 β**: BKZ-95 → β=105 pump。更高的 blocksize 意味着更强的局部约化，向量 ℓ2 范数从 ~111 降低，ℓ∞ 概率性降至 16
2. **充分的随机探索**: 40 个独立随机种子 × 2160 次 pump 调用 = 巨大的搜索多样性
3. **聚焦策略**: 只 pump 影响最优行的顶部 block，避免浪费时间在无关 block 上
4. **硬件优势**: 384 核 EPYC 服务器提供了传统 4-8 核机器无法企及的并行能力

### ℓ2 与 ℓ∞ 的关系

在 240 维中，若向量元素服从 N(0, σ²)：
- σ ≈ 7.1 时 ℓ2 ≈ 111（BKZ-95 基的状态）
- P(ℓ∞ ≤ 16) ≈ 0.986^240 ≈ 3.4%
- β=105 pump 后 ℓ2 降低，等效 σ 减小，P(ℓ∞ ≤ 16) 显著增加

## 文件说明

| 文件 | 用途 |
|------|------|
| `parallel_pump.py` | 大规模并行 pump 求解器（40 workers） |
| `focused_pump.py` | 单链聚焦顶部 block 的 pump 求解器 |
| `bkz_patched.py` | G6K bkz.py 补丁（LLL try/except） |
| `pump_patched.py` | G6K pump.py 补丁（LLL try/except） |
| `p3_solution.txt` | 最终解向量 (v, u) 文本格式 |
| `p3_solved_final.npy` | 最终解向量 NumPy 格式 |

## 运行方法

```bash
# 1. 安装依赖
pip install numpy fpylll g6k

# 2. 打 G6K LLL 补丁
python3 -c "
import g6k.algorithms.bkz, g6k.algorithms.pump, os
# 将 g6k.lll() 调用包裹 try/except（或手动复制 bkz_patched.py, pump_patched.py）
"

# 3. 放置 BKZ-95 预约化基（g6k_push_b95_p3.npy）

# 4. 运行
python3 parallel_pump.py --workers 40 --threads 8 --beta-start 90 --beta-end 130
```

## 关键注意事项

1. **不要用 GPU-Tensor fork** — 在 BKZ context switch 时崩溃
2. **不要用 fpylll BKZ 直接跑 SIS 基** — Babai 死循环
3. **G6K LLL 调用必须 try/except 包裹** — SIS 基上 Babai 会崩溃
4. **每次 pump 创建新 Siever 实例** — 确保种子独立性
5. **不要堆叠 C++ 补丁** — 纯 Python 层的 pump 策略足够

---

*求解日期: 2026-06-05*
*作者: G6K High-BS Pump Method + 384-Core Parallel Exploration*
