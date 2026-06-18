# P4: Inhomogeneous SIS (CVP), q=120, N=240, target linf≤16

## Best result: linf=18
- File: `p4_best_linf18.npy`
- Method: Numba-accelerated Klein CVP on hybrid basis (BKZ-100 + homo=16 + deep LLL δ=0.9999)
- Speed: ~6500 trials/s

## Best basis: hybrid (homo linf=16, d50→100=0.376)
- File: `p4_best_basis_h16.npy`
- Creation: BKZ-100 basis, row0 replaced with homo=16 vector from BKZ-110, then LLL δ=0.9999
- GS profile smoother than BKZ-100 (d50→100=0.148)

## Key scripts
- `p4_klein_numba.py` — Numba JIT Klein worker (6500/s)
- `p4_klein_numba_h16.py` — Numba Klein on hybrid basis
- `p4_gpu_klein.py` — GPU-batched Klein (slower, ~531/s)
- `p4_enum_b16.py` — Beam search enumeration on homo=16 basis
- `test_p4_gpu_fixed.py` — GPU G6K BKZ test on P4

## Key findings
- GS smoothness (d50→100 ratio) > homo linf for Klein CVP
- Numba JIT gives 16x speedup over pure Python
- GPU BKZ crashes on multi-block tour (C++ bugs fixed later for P6)
- Kannan embedding (CVP→SVP) ineffective for l∞ norm
- Target formula: target[M:] = (-t) as negative float
