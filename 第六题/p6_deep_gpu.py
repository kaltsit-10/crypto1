#!/usr/bin/env python3
"""P6: Deep LLL + GPU G6K progressive BKZ."""
import numpy as np, os, sys, time
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor')
sys.path.insert(0, '/home/linux/G6K-GPU-Tensor/g6k/algorithms')
os.environ['LD_LIBRARY_PATH'] = '/home/linux/G6K-GPU-Tensor/kernel:' + os.environ.get('LD_LIBRARY_PATH','')

from fpylll import IntegerMatrix, GSO, LLL
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.utils.stats import dummy_tracer

Q,M,N=140,140,280;HALF=70;GAMMA=17
SAVE_DIR='/home/linux/PycharmProjects/pythonProject/crypto1/p6_saves'
os.makedirs(SAVE_DIR,exist_ok=True)

# Load best saved basis and do deep LLL
best_file = os.path.join(SAVE_DIR, 'p6_gpu_bkz60.npy')
if os.path.exists(best_file):
    B = np.load(best_file).astype(np.int64)
    print(f'Loaded BKZ-60 basis, running deep LLL (delta=0.9999)...', flush=True)
else:
    with open('/home/linux/PycharmProjects/pythonProject/crypto1/p6优化基/p6_basis_bkz40.txt') as f:
        lines=[l for l in f.readlines() if not l.startswith('#')]
    B=np.zeros((N,N),dtype=np.int64)
    for i in range(N):
        B[i]=np.array(list(map(int,lines[i+1].split())),dtype=np.int64)
    print('Loaded BKZ-40 basis, running deep LLL...', flush=True)

# Deep LLL via GSO
m = IntegerMatrix(N, N)
for i in range(N):
    for j in range(N): m[i, j] = int(B[i, j])
mg_lll = GSO.Mat(m, U=IntegerMatrix.identity(N), UinvT=IntegerMatrix.identity(N))
mg_lll.update_gso()
lll = LLL.Reduction(mg_lll, delta=0.9999)
lll()
mg_lll.update_gso()

B_deep = np.zeros((N, N), dtype=np.int64)
for i in range(N):
    for j in range(N): B_deep[i, j] = int(m[i, j])

# Check linf after deep LLL
def best_linf(Bn):
    bl,br=999,0
    for i in range(N):
        row=Bn[i];v=row[:M];u=row[M:]%Q
        u=np.where(u>=HALF,u-Q,u).astype(np.int64)
        if np.all(v==0) and np.all(u==0): continue
        li=max(int(np.abs(v).max()),int(np.abs(u).max()))
        if li<bl: bl,br=li,i
    return br,bl

br,bl=best_linf(B_deep)
print(f'Deep LLL done: linf={bl}', flush=True)
np.save(os.path.join(SAVE_DIR,'p6_deep_lll.npy'), B_deep)

# GSO check
Bf=B_deep.astype(np.float64)
gso_r=np.zeros(N)
for i in range(N):
    gso_r[i]=np.linalg.norm(Bf[i])**2
print(f'GS: [0]={gso_r[0]:.0f} [50]={gso_r[50]:.0f} [100]={gso_r[100]:.0f} [200]={gso_r[200]:.0f}')
print(f'd50->100={np.sqrt(gso_r[100]/gso_r[50]):.4f}', flush=True)

def make_gso(Bn):
    m=IntegerMatrix(N,N)
    for i in range(N):
        for j in range(N): m[i,j]=int(Bn[i,j])
    mg=GSO.Mat(m,U=IntegerMatrix.identity(N),UinvT=IntegerMatrix.identity(N))
    mg.update_gso()
    return mg

B_curr=B_deep.copy()
best_so_far=bl
start_beta=65

# Resume from best saved basis if available
for b in [90,85,80,75,70,65]:
    f=os.path.join(SAVE_DIR,f'p6_deep_bkz{b}.npy')
    if os.path.exists(f):
        B_curr=np.load(f).astype(np.int64)
        br,bl=best_linf(B_curr)
        best_so_far=bl
        start_beta=b+5
        print(f'Resumed from BKZ-{b}: linf={bl}', flush=True)
        break

for beta in range(start_beta, 95, 5):
    if best_so_far<=GAMMA: break
    print(f'GPU BKZ-{beta}...', flush=True)
    t0=time.time()
    mg=make_gso(B_curr)
    g=Siever(mg,SieverParams(threads=16,default_sieve='gauss'),seed=int(time.time()))
    try:
        pump_n_jump_bkz_tour(g, dummy_tracer, beta,
                            pump_params={'down_sieve':False}, verbose=True)
    except Exception as e:
        print(f'  Error: {e}', flush=True)
        continue
    dt=time.time()-t0
    B_curr=np.zeros((N,N),dtype=np.int64)
    for i in range(N):
        for j in range(N): B_curr[i,j]=int(g.M.B[i,j])
    br,bl=best_linf(B_curr)
    print(f'  {dt:.0f}s, linf={bl} (best={best_so_far})', flush=True)
    if bl<best_so_far:
        best_so_far=bl
        np.save(os.path.join(SAVE_DIR,f'p6_deep_bkz{beta}.npy'),B_curr)
    if bl<=GAMMA:
        print(f'*** P6 SOLVED! linf={bl} ***', flush=True)
        row=B_curr[br];v=row[:M];u=row[M:]%Q
        u=np.where(u>=HALF,u-Q,u).astype(np.int64)
        np.save(os.path.join(SAVE_DIR,'p6_solution.npy'),np.concatenate([v,u]))
        break

print(f'Final: linf={best_so_far}', flush=True)
