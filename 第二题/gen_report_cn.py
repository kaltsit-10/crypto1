#!/usr/bin/env python3
"""Generate Chinese PDF report using fpdf2."""
from fpdf import FPDF
import os

BASE = '/home/dys1013/crypto_challenge/results_p2'
FONT = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'

class PDF(FPDF):
    def header(self):
        self.set_font('Droid', '', 10)
        self.cell(0, 6, 'Problem 2 SIS CVP 求解报告', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def section_title(self, title):
        self.set_font('Droid', '', 13)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 9, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def body(self, text):
        self.set_font('Droid', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def my_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
        self.set_font('Droid', '', 9)
        # Header
        self.set_fill_color(220, 220, 220)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6.5, str(cell), border=1, align='C')
            self.ln()
        self.ln(2)

pdf = PDF()
pdf.add_font('Droid', '', FONT)
pdf.set_auto_page_break(True, 15)

# ─── Page 1 ───
pdf.add_page()

pdf.section_title('一、问题描述')
pdf.body('寻找 v, u ∈ Z^100，满足：')
pdf.body('    Av + u ≡ t (mod 100),   ||v||_∞ ≤ 15,  ||u||_∞ ≤ 15')
pdf.body('其中 A ∈ Z^{100×100}_{100}，t ∈ Z^{100}_{100}。格维度 200，模数 q=100。')

pdf.section_title('二、所有尝试方法汇总')
pdf.body('BKZ 渐进归约结果：')

pdf.my_table(
    ['BKZ级别', '耗时', '齐次最佳', 'Babai CVP'],
    [
        ['40',  '3s',    'score=32', 'score=45'],
        ['50',  '15s',   '—',        '—'],
        ['60',  '50s',   'score=30', 'score=45'],
        ['65',  '2min',  'score=21', 'score=31'],
        ['70',  '7min',  'score=17', 'score=30'],
        ['75',  '1.5h',  'score=15', 'score=24'],
        ['80',  '3.4h',  'score=14 ✓', 'score=24'],
        ['85',  '15.8h', '已杀',      '已杀'],
    ],
    [30, 20, 30, 30]
)

pdf.body('BKZ-80 是关键：齐次 ℓ∞=14（3行达标），但 Babai CVP 卡在 24。')

pdf.body('所有方法完整对比：')
pdf.my_table(
    ['方法', '最佳Score', '结论'],
    [
        ['Babai CVP (BKZ-80)',          '24', '基线'],
        ['Babai + 短齐次行组合',          '21', '局部最优'],
        ['随机化 Babai (5万次)',          '24', '无改善'],
        ['模拟退火',                      '24', '随机扰动破坏性大'],
        ['贪心坐标下降',                   '22', '近视，卡死'],
        ['定向行组合搜索',                 '21', '子空间不匹配'],
        ['LLL+Kannan (修正u公式)',         '44', '远差于Babai'],
        ['G6K workout (bs=40-80)',      '24-26', 'CVP无增益'],
        ['fpylll CVP.closest_vector',   '—', '3.8h无结果'],
        ['Z3 SMT / CP-SAT',             '—', '超时'],
    ],
    [55, 25, 35]
)

# ─── Page 2 ───
pdf.add_page()
pdf.section_title('三、成功解法：B&B GSO剪枝枚举')

pdf.body('第1步：BKZ-80格基归约（3h23m）')
pdf.body('构造正确的SIS格基（200×200）：')
pdf.body('    B = [ I_100,  -A^T ]')
pdf.body('        [ 0,      q·I_100 ]')
pdf.body('用fpylll BKZ-80归约（strategies=default.json, max_loops=2）。')

pdf.body('第2步：Babai CVP → 初始解 score=24')
pdf.body('在BKZ-80基上做Babai最近平面算法，得到初始候选解。')

pdf.body('第3步：B&B GSO剪枝枚举 → score=15（关键突破）')
pdf.body('原理：Babai取整误差集中在最后几个GSO维度。BKZ-80基中：')
pdf.body('  · 维度 0-191：GSO范数大（~10⁴），Babai取值准确')
pdf.body('  · 维度 192-199：GSO范数小，但μ系数将修正反向传播到全部200个坐标')
pdf.body('')
pdf.body('做法：')
pdf.body('  1. 获取Babai系数 c_j (j=0..199)')
pdf.body('  2. 枚举最后8个系数的邻域，每个 ±4 → 9⁸ ≈ 4300万组合')
pdf.body('  3. 对每个组合计算v，通过 u=centered((t-Av)%q) 得到u')
pdf.body('  4. GSO ℓ₂下界剪枝：若残差ℓ₂ > 当前最优，跳过')
pdf.body('  5. 剩余192维仍用Babai取整')

pdf.body('求解路径：24 → 23 → 22 → 20 → 18 → 17 → 16 → 15')
pdf.body('耗时：约6分钟。43M组合评估完毕。')
pdf.body('')
pdf.body('最终结果：||v||_∞=15, ||u||_∞=15, (Av+u-t)%100=0 ✓')

# ─── Page 3 ───
pdf.add_page()
pdf.section_title('四、为什么B&B比其他所有方法快几个数量级')

pdf.body('1. 搜对了空间')
pdf.body('Babai误差几乎全部集中在最后8个GSO维度。前192维的GSO范数大，Babai在这些维度的取整误差对最终解影响可忽略。后8维虽然GSO范数小，但其μ系数将系数变化传播到全部坐标。修正这8个维度±4就足以将ℓ∞从24压到15。')
pdf.body('')

pdf.body('2. GSO ℓ₂剪枝准确高效')
pdf.body('ℓ∞剪枝（之前buggy版本的做法）在根节点就剪掉一切——因为partial v包含前几行的大坐标值。GSO ℓ₂下界准确反映了剩余改进空间，对归约好的格基剪枝效果极佳。')
pdf.body('')

pdf.body('3. 逐方法对比')
pdf.my_table(
    ['失败方法', 'B&B为什么胜过它'],
    [
        ['模拟退火', '随机扰动无结构引导，99.9%的步长炸分。B&B只探GSO结构化的扰动'],
        ['贪心下降', '逐坐标贪心是近视的，无法发现多系数协调修正。B&B同时搜8维'],
        ['短齐次行', '3条短行(ℓ∞≤15)的大坐标位置与Babai超标坐标不重叠——在格的不同方向'],
        ['fpylll CVP', '搜全部200维——搜索树巨大，4h无果。B&B只搜8维'],
        ['G6K workout', '改善格基，不直接解决CVP。Babai已是局部最优'],
        ['LLL+Kannan', 'CVP距离>λ₁，c=-1解永不最短'],
    ],
    [40, 80]
)

pdf.body('')
pdf.body('4. 核心洞察')
pdf.body('CVP解离Babai解并不远——ℓ∞差距仅24→15。但这个差距集中在最后几个GSO维度，小系数变化通过μ矩阵产生大范围全局效应。200维全枚举是杀鸡用牛刀；8维±4加GSO剪枝，几分钟就能找到。')
pdf.body('')

pdf.section_title('五、结论')
pdf.body('1. BKZ-80归约（3.4h）——在正确SIS格基上的必要预处理')
pdf.body('2. Babai CVP——基线解(score=24)，已是局部最优，小扰动无法改进')
pdf.body('3. B&B GSO剪枝（6min）——枚举最后8个GSO系数±4，ℓ₂剪枝，突破局部最优到score=15')
pdf.body('')
pdf.body('正确的搜索子空间 + 正确的剪枝度量 = 分钟级求解。')

# Save
path = os.path.join(BASE, 'report_cn.pdf')
pdf.output(path)
print(f'Saved: {path}', flush=True)
