# P6: Homogeneous SIS, q=140, N=280, target linf≤17

## Best result: linf=59 (BKZ basis)
- File: `p6_basis_bkz60.npy` (280×280 basis, best row linf=59)
- Method: Deep LLL δ=0.9999 + GPU G6K progressive BKZ β=45→90
- G6K-GPU-Tensor with 3 C++ fixes applied

## G6K-GPU fixes (for N>38 BKZ)
1. `kernel/gpu_sieve.cpp` — guard empty bucketcenters in nth_element
2. `g6k/siever.pyx` — Python 3 compatibility (None → -inf in insert_best_lift)
3. `kernel/control.cpp` — guard i<ll in get_lift_bound

## Key scripts
- `p6_deep_gpu.py` — Deep LLL + GPU BKZ with linf sorting + CPU gauss
- `p6_gpu_full.py` — Full GPU BKZ with all 3 fixes
- `p6_g6k_linf.py` — linf-sorted CPU gauss BKZ
- `test_p6_gpu_single.py` — Single-block GPU BKZ test

## Current status
- BKZ-90 running, best linf=59
- Need β=100-120 to reach linf≤17 (ref: P3 needed β=95 for N=240 to reach 18)
- GPU utilization low (~5%), bottleneck is CPU LLL/GSO
