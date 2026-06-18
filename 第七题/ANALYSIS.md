# 第七题分析与三种求解方案

## 问题参数

| 参数 | 值 |
|------|-----|
| n = m = q | 140 |
| γ（ℓ∞ 上界） | 17 |
| t 向量 | **非零**（非齐次 SIS），t ∈ [0, 139]^140 |
| 额外约束 | 无 |

**目标**：找 (u, v) ∈ ℤ^140 × ℤ^140，满足：
1. A·v + u ≡ t (mod 140)
2. ℓ∞(u, v) ≤ 17

**与其他题的关键区别**：
- vs P3（n=120，t=0）：维度更大（140 vs 120），且 **t ≠ 0**（非齐次）
- vs P5（n=120，t=0，额外ℓ₂条件）：无额外ℓ₂条件，但 t ≠ 0
- vs P6（n=140，t=0）：完全相同规模，但本题 t ≠ 0

---

## 核心数学分析

### 格结构

SIS 齐次格 L₀（维度 280，det = 140^140）：
```
GH(280) ≈ √(280/(2πe)) × √140 ≈ 47.9
```

非齐次解集 L_t = {(v,u) : Av + u ≡ t (mod 140)}，是 L₀ 的一个陪集。

**直接尝试 u₀ = t（特解）**：
- t 的范围是 [0, 139]，ℓ∞(u₀) = max(t) = 139 >> 17，**不可用**
- 需要找到陪集 L_t 中 ℓ∞ ≤ 17 的点

### 转化为近似 CVP

在 L₀ 中寻找最接近目标向量 w₀ = (0, ..., 0, t₁, ..., t_n)^T 的格点，
其差值 (v, u-t) 即满足 Av + (u-t) ≡ 0 (mod q) 且期望 ℓ₂ ≈ GH ≈ 47.9。

### ℓ∞ 成功概率

对 ℓ₂ ≈ GH ≈ 47.9、dim=280 的向量，P(ℓ∞ ≤ 17) ≈ **0.9999**（几乎确定满足！）

这意味着：只要能找到陪集中 ℓ₂ 足够小（≈GH）的向量，ℓ∞ 条件**自动满足**。
**P7 不需要 P5 那种特殊处理，标准格算法足够。**

---

## 方案一：Kannan 嵌入 + G6K 筛法（直接改造 P3 代码）

### 核心思想

Kannan 嵌入将非齐次 CVP 转化为齐次 SVP：

**构造 (2n+1) × (2n+1) = 281 × 281 的增广格基**：

```
行 0..n-1:  [ q·I_n |  A    |  0 ]   ← 原 SIS 格的前 n 行
行 n..2n-1: [  0    | -I_n  |  0 ]   ← 原 SIS 格的后 n 行
行 2n:      [  0    |  t^T  |  c ]   ← 新增行，c = γ = 17
```

**核心性质**：若增广格中存在短向量 w = (v, u-t, 1)（最后分量 = 1），则：
- A·v + u ≡ t (mod q)（SIS 方程满足）
- w 的前 2n 个分量即为 (v, u-t)，第 2n+1 个分量 = 1
- 由于 c = γ，短向量条件 ‖w‖∞ ≤ γ 要求前 2n 个分量 ℓ∞ ≤ γ

### 方法步骤

```python
# 1. 构造 281×281 Kannan 格基
B = zeros(281, 281)
B[0:n, 0:n] = q * I_n          # 左上 q·I
B[0:n, n:2n] = A                # 右上 A
B[n:2n, n:2n] = -I_n            # 右下 -I
B[2n, n:2n] = t                 # t 向量填入最后行
B[2n, 2n] = gamma               # Kannan 缩放因子 c = 17

# 2. LLL 预处理
LLL.reduction(B)

# 3. BKZ-95 预约化（与 P3 相同）
BKZ.reduction(B, BKZ.Param(block_size=95, max_loops=4, AUTO_ABORT))

# 4. G6K pump (parallel, 40 workers × 8 threads)
# 与 P3 完全相同的 parallel_pump.py 框架

# 5. 检验：对每个找到的短行 row：
def check_kannan_row(row):
    last = row[2*n]          # 最后一个坐标
    if abs(last) != gamma:
        return False         # 不是目标向量
    sign = last // gamma     # ±1
    u_minus_t = sign * row[n:2*n]   # 对应的 u-t 部分
    v = sign * row[0:n]              # v 部分
    u = u_minus_t + t               # 恢复 u = (u-t) + t
    # 验证：A·v + u ≡ t (mod q)，且 ℓ∞(v,u) ≤ γ
    return verify(v, u)
```

### 时间估计

- dim=281 比 P3 的 dim=240 大 17%，但 Kannan 嵌入后难度类似
- P3 用 7.5 分钟（446秒）解决 dim=240
- P7 估计 **30-120 分钟**（dim 增大 + Kannan 嵌入额外复杂度）
- 40 workers × 8 threads = 320 线程（与 P3 相同）

---

## 方案二：BKZ 约化 + Babai 近似 CVP（快速尝试）

### 核心思想

先对齐次 SIS 格做 BKZ 约化，再用 Babai 最近平面算法（Nearest Plane）
在约化基上找到最接近目标的格点，得到陪集代表元。

### 方法步骤

**第一阶段：BKZ 约化 SIS 格（dim=280）**
```python
B_sis = build_sis_basis(A)    # 280×280
LLL.reduction(B_sis)
BKZ.reduction(B_sis, BKZ.Param(block_size=60))  # BKZ-60，较快
```

**第二阶段：Babai 最近平面**
```python
from fpylll import FPLLL, IntegerMatrix, GSO, LLL

M = GSO.Mat(B_fp)
M.update_gso()

# 目标向量：w = (0,...,0, t[0],...,t[n-1]) in Z^(2n)
target = [0]*n + list(t)

# Babai 最近平面（fpylll 内置）
from fpylll.algorithms.babai import babai_nearest_plane
v_babai = babai_nearest_plane(M, target)
```

**第三阶段：验证和改进**
```python
close_pt = np.array(v_babai, dtype=np.int64)
# close_pt 是格中最近的点，差值 = target - close_pt
diff = np.array(target) - close_pt

# diff 的前 n 个分量是 v，后 n 个是 u-t
v = diff[:n]
u_minus_t = diff[n:]
u = (u_minus_t + t) % q  # 取模后中心化到 [-q/2, q/2]
u = ((u + q//2) % q) - q//2

linf = max(np.max(np.abs(v)), np.max(np.abs(u)))
if linf <= gamma:
    print("DONE!")  # Babai 直接给出解
else:
    print(f"Babai ℓ∞ = {linf}, need further reduction")
    # 接下来继续提高 BKZ 块大小重试
```

### 成功概率分析

- BKZ-60 约化后，Babai 近似误差 ≈ δ^n × GH，随 β 增大而减小
- BKZ-90 时：δ_90 ≈ 1.0096，GH=47.9
  - Babai 误差 ≈ 1.0096^140 × 47.9 ≈ 161（太大）
- BKZ-120 时：δ_120 ≈ 1.0084，近似更准
  - 理论上 BKZ-120 后 Babai 误差 ≈ GH ≈ 47.9，可能刚好满足 ℓ∞ ≤ 17

**实际上**：由于 P(ℓ∞≤17 | ℓ₂≈GH)≈1，BKZ-100 之后 Babai 成功的可能性很高。

### 优点

- 不需要 Kannan 嵌入（少一维）
- Babai 算法本身极快（毫秒级）
- 失败时可逐步提高 BKZ β

---

## 方案三：G6K otf_lift 非齐次模式（最优雅）

### 核心思想

G6K 的 `otf_lift=True` 参数实际上就是为了处理非齐次 SIS 问题（CVP）设计的。
P3 代码中已经用了 `otf_lift=True`，对非齐次情况只需修改 SIS 格基构造。

### 方法步骤

```python
# 构造含 t 的 SIS 格基：将 t 作为格基一部分
# 方法：在 B[:n, :] 中加入 t 的影响
# 直接在格基的最后一列/行嵌入 t

# 等价做法：构造对偶格，令 u₀ = t (mod q) 折叠到 [-q/2, q/2]
# 然后作为"lift offset"传入 G6K

from g6k import Siever, SieverParams

# 将 t 中心化到 [-q/2, q/2]
t_centered = np.array([(ti if ti <= q//2 else ti - q) for ti in t])

# 构造非齐次 SIS 的增广格基（Kannan，dim=281）
B_kannan = build_kannan_basis(A, t_centered, gamma=17)

# 其余与 P3 的 parallel_pump.py 完全相同
# 在检验时改为：
def check_row(row):
    if abs(row[280]) != 17:
        return False
    # ... 还原 (v, u) 并验证
```

### 与 P3 代码的差异

| 项目 | P3 | P7 |
|------|----|----|
| 格维度 | 240 | 281 |
| 格基构造 | `build_sis_basis(A)` | `build_kannan_basis(A, t, γ)` |
| 检验函数 | `Av + u ≡ 0` | `Av + u ≡ t`，且最后分量 = ±γ |
| BKZ β | 95 | 95-105 |
| 预期时间 | 446s | ~1-4h |

---

## 三种方案对比

| 方案 | 核心方法 | 难度 | 预期时间 | 代码复用 |
|------|---------|------|---------|---------|
| 方案一：Kannan + G6K pump | 增广格 + 筛法 | 中 | 1-4h | P3 小改 |
| 方案二：BKZ + Babai CVP | 格约化 + 近似最近格点 | 低 | 10-60min | 纯 fpylll |
| 方案三：G6K otf_lift | G6K 内置 CVP 支持 | 中 | 1-4h | P3 几乎直接用 |

### 推荐执行顺序

1. **先跑方案二**（最快：BKZ-100 + Babai，可能 20 分钟内出结果）
2. **同时启动方案一**（最可靠：Kannan 是标准方法）
3. **方案三作为备用**（与方案一本质相同，代码更简洁）

---

## 验证标准

```python
def verify_p7(v, u, A, t, q=140, gamma=17):
    # 条件1: 非齐次 SIS 方程
    residual = (A @ v + u - t) % q
    assert np.all(residual == 0), f"SIS violated"
    # 条件2: 无穷范数
    linf = max(np.max(np.abs(v)), np.max(np.abs(u)))
    assert linf <= gamma, f"ℓ∞ = {linf} > {gamma}"
    print(f"✅ Valid! ℓ∞={linf}")
```

---

## 文件结构

```
第七题/
├── ANALYSIS.md          ← 本文件
├── method1_kannan.py    ← 方案一：Kannan 嵌入 + G6K pump
├── method2_babai.py     ← 方案二：BKZ + Babai CVP（最快）
├── method3_g6k_lift.py  ← 方案三：G6K otf_lift 直接 CVP
└── verify.py            ← 通用验证脚本
```
