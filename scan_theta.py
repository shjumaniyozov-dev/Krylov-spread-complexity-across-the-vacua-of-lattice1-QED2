import numpy as np, time, sys
from fractions import Fraction as F
from multiprocessing import Pool
from krylov_run import *
from scipy.linalg import eigh_tridiagonal
kind=sys.argv[1]; N=int(sys.argv[2]); x=sys.argv[3]; mu=sys.argv[4]; tag=sys.argv[5]
ls=[F(k,100) for k in range(0,100)]
ts=np.linspace(0,4000,8001)
def run(l):
    S,P=model(kind,N)
    nmax=S.D; bits=int(2.6*nmax)+600
    t0=time.time()
    a,b=coeffs(kind,N,x,mu,str(l),nmax,bits=bits)
    C,Pn,tail=complexity(a,b,ts)
    ev,U=eigh_tridiagonal(a,b)
    pinf=(U**2)@(U[0,:]**2)          # infinite-time average p_n (non-degenerate spectrum)
    Cinf=float(np.arange(len(a))@pinf)
    return dict(l=float(l),a=a,b=b,C=C,Cinf=Cinf,depth=len(a),time=time.time()-t0)
if __name__=="__main__":
    with Pool(2) as pool:
        out=pool.map(run,ls)
    np.savez_compressed(f"/home/claude/fdp/data/scan_{tag}.npz",ts=ts,l=np.array([o['l'] for o in out]),
        C=np.array([o['C'] for o in out]),Cinf=np.array([o['Cinf'] for o in out]),
        depth=np.array([o['depth'] for o in out]),
        a=np.array([np.pad(o['a'],(0,4000-len(o['a']))) for o in out]),b=np.array([np.pad(o['b'],(0,4000-len(o['b']))) for o in out]))
    for o in out: print("%.2f depth=%d Cinf=%.4f mean=%.4f t=%.1f"%(o['l'],o['depth'],o['Cinf'],o['C'].mean(),o['time']))
