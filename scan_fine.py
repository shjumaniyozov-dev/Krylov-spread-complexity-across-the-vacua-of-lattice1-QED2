# fine scan in the false-vacuum window, windowed time averages, N given
import numpy as np, time, sys
from fractions import Fraction as F
from multiprocessing import Pool
from krylov_run import *
from scipy.linalg import eigh_tridiagonal
kind=sys.argv[1]; N=int(sys.argv[2]); x=sys.argv[3]; mu=sys.argv[4]; tag=sys.argv[5]
k0,k1,den=int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8])
ls=[F(k,den) for k in range(k0,k1+1)]
ts=np.linspace(0,1000,2001)
def run(l):
    S,P=model(kind,N)
    nmax=S.D; bits=int(2.6*nmax)+600
    a,b=coeffs(kind,N,x,mu,str(l),nmax,bits=bits)
    C,Pn,tail=complexity(a,b,ts)
    ev,U=eigh_tridiagonal(a,b); pinf=(U**2)@(U[0,:]**2)
    return float(l),C,float(np.arange(len(a))@pinf),len(a)
if __name__=="__main__":
    with Pool(2) as pool: out=pool.map(run,ls)
    np.savez_compressed(f"/home/claude/fdp/data/fine_{tag}.npz",ts=ts,l=np.array([o[0] for o in out]),C=np.array([o[1] for o in out]),Cinf=np.array([o[2] for o in out]),depth=np.array([o[3] for o in out]))
    print("done",len(out))
