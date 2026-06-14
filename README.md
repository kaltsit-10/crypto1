# P4 Solution — Inhomogeneous SIS (CVP)

**Problem**: Find (v,u) with A·v + u ≡ t (mod 120), ℓ∞ ≤ 16
**Dimensions**: 240 (120×120 matrix A, q=120)

## Current Best
- **Inhomogeneous linf: 18** (|v|=18, |u|=18)
- **Homogeneous linf: 17** (basis row 0)
- **Target: 16** (gap: 2)

## Approach
- G6K GPU-Tensor BKZ reduction (blocksizes up to b=140)
- Klein randomized rounding CVP (Babai nearest plane + Gaussian perturbation)
- Multi-worker parallel search with individual save files

## Files
- `solutions/best_linf18.npy` — Best inhomogeneous solution (linf=18)
- `solutions/k*.npy` — Individual worker bests
- `scripts/p4_klein_worker.py` — Main Klein CVP worker
- `scripts/p4_bkz_high.py` — GPU BKZ reduction script
- `bases/homo_linf17.npy` — Best homogeneous basis
- `bases/bkz100_linf21.npy` — BKZ-100 basis (best for Klein CVP)
- `problem4.txt` — Problem definition (A matrix, t vector)

## Key Finding
BKZ-100 basis (homogeneous linf=21) gives BETTER Klein CVP results than BKZ-130 (homogeneous linf=17) because of more gradual GS profile. The GS cliff at indices 50-100 determines Klein precision.

## Running
```bash
# Single worker
python3 p4_klein_worker.py <seed> <save_path>

# GPU BKZ
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 p4_bkz_high.py
```
