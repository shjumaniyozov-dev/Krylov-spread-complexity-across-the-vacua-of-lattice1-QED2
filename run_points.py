import numpy as np, time, sys
from fractions import Fraction as F
from multiprocessing import Pool
from krylov_run import *
from scipy.linalg import eigh_tridiagonal
kind=sys.argv[1]; N=int(sys.argv[2]); x=sys.argv[3]; mu=sys.argv[4]; tag=sys.argv[5]; ls=sys.argv[6].split(',')
nproc=int(sys.argv[7]) if len(sys.argv)>7 else 2
ts=np.linspace(0,1000,2001)
def run(l):
    S,P=model(kind,N)
    nmax=S.D; bits=int(2.6*nmax)+600
    t0=time.time()
    a,b=coeffs(kind,N,x,mu,l,nmax,bits=bits)
    C,Pn,tail=complexity(a,b,ts)
    ev,U=eigh_tridiagonal(a,b); pinf=(U**2)@(U[0,:]**2)
    print(l,"depth",len(a),"Cinf",float(np.arange(len(a))@pinf),"win200",C[ts<=200].mean(),"time",time.time()-t0,flush=True)
    return float(F(l)),C,float(np.arange(len(a))@pinf),a,b
if __name__=="__main__":
    with Pool(nproc) as pool: out=pool.map(run,ls)
    np.savez_compressed(f"/home/claude/fdp/data/pts_{tag}.npz",ts=ts,l=np.array([o[0] for o in out]),C=np.array([o[1] for o in out]),Cinf=np.array([o[2] for o in out]),
        a=np.array([np.pad(o[3],(0,8000-len(o[3]))) for o in out]),b=np.array([np.pad(o[4],(0,8000-len(o[4]))) for o in out]),depth=np.array([len(o[3]) for o in out]))
