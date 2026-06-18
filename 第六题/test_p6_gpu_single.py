#!/usr/bin/env python3
"""P6: GPU G6K single-block BKZ (jump=1000 to avoid cross-block crash)."""
import numpy as np, os, sys, time
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor')
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor/g6k/algorithms')
os.environ['LD_LIBRARY_PATH'] = '/home/linux/G6K-GPU-Tensor/kernel:' + os.environ.get('LD_LIBRARY_PATH','')
from fpylll import IntegerMatrix, GSO
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.utils.stats import dummy_tracer

Q,M,N=140,140,280;HALF=70;GAMMA=17
SAVE_DIR='/home/linux/PycharmProjects/pythonProject/crypto1/p6_saves'
os.makedirs(SAVE_DIR,exist_ok=True)

with open('/home/linux/PycharmProjects/pythonProject/crypto1/p6优化基/p6_basis_bkz40.txt') as f:
    lines=[l for l in f.readlines() if not l.startswith('#')]
B=np.zeros((N,N),dtype=np.int64)
for i in range(N):
    B[i]=np.array(list(map(int,lines[i+1].split())),dtype=np.int64)

def best_linf(Bn):
    bl,br=999,0
    for i in range(N):
        row=Bn[i];v=row[:M];u=row[M:]%Q
        u=np.where(u>=HALF,u-Q,u).astype(np.int64)
        if np.all(v==0) and np.all(u==0): continue
        li=max(int(np.abs(v).max()),int(np.abs(u).max()))
        if li<bl: bl,br=li,i
    return br,bl

def make_gso(Bn):
    m=IntegerMatrix(N,N)
    for i in range(N):
        for j in range(N): m[i,j]=int(Bn[i,j])
    mg=GSO.Mat(m,U=IntegerMatrix.identity(N),UinvT=IntegerMatrix.identity(N))
    mg.update_gso()
    return mg

B_curr=B.copy()
br,bl=best_linf(B_curr)
best_so_far=bl
print(f'P6 GPU single-block: start linf={bl}', flush=True)

for beta in [45,50,55,60]:
    if best_so_far<=GAMMA: break
    print(f'GPU BKZ-{beta} (single block)...', flush=True)
    t0=time.time()
    mg=make_gso(B_curr)
    p=SieverParams(threads=4)
    g=Siever(mg,p,seed=0)
    try:
        pump_n_jump_bkz_tour(g, dummy_tracer, beta,
                            pump_params={'down_sieve':False},
                            jump=1000, verbose=True)
    except Exception as e:
        print(f'  GPU failed: {e}, fallback to CPU...', flush=True)
        mg2=make_gso(B_curr)
        p2=SieverParams(threads=8, default_sieve='gauss')
        g2=Siever(mg2,p2,seed=0)
        try:
            pump_n_jump_bkz_tour(g2, dummy_tracer, beta,
                                pump_params={'down_sieve':False},
                                jump=1000, verbose=True)
            g=g2
        except Exception as e2:
            print(f'  CPU also failed: {e2}', flush=True)
            continue
    dt=time.time()-t0
    B_curr=np.zeros((N,N),dtype=np.int64)
    for i in range(N):
        for j in range(N): B_curr[i,j]=int(g.M.B[i,j])
    br,bl=best_linf(B_curr)
    print(f'  {dt:.0f}s, linf={bl} (best={best_so_far})', flush=True)
    if bl<best_so_far:
        best_so_far=bl
        np.save(os.path.join(SAVE_DIR,f'p6_gpu_bkz{beta}.npy'),B_curr)
    if bl<=GAMMA:
        print(f'*** SOLVED! ***', flush=True)
        break

print(f'Final: {best_so_far}', flush=True)
