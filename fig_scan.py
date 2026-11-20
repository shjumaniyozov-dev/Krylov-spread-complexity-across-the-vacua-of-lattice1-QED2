import numpy as np
from fig_common import *
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for tag,N,c,ls in [("full8",8,S1,"-"),("full10",10,S2,"--"),("full12",12,S3,"-.")]:
    d=np.load(f"../data/scan_{tag}.npz"); l=d["l"]; ts=d["ts"]
    w=d["C"][:,ts<=200].mean(1)
    ax[0].plot(l,w,ls,color=c,label=f"$N={N}$")
    ax[1].semilogy(l,d["Cinf"],ls,color=c,label=f"$N={N}$")
for a in ax:
    a.axvline(0.5,color="#8a8984",lw=0.8,ls=":")
    a.set_xlabel(r"$\theta/2\pi=\ell$"); a.grid(color=GRID,lw=0.5)
ax[0].set_ylabel(r"$\overline{C}_{[0,200]}$"); ax[1].set_ylabel(r"$\overline{C}_\infty$")
ax[0].legend(frameon=False,loc="upper left"); 
ax[0].text(0.02,0.97,"(a)",transform=ax[0].transAxes,va="top",ha="left",fontsize=9); ax[1].text(0.02,0.97,"(b)",transform=ax[1].transAxes,va="top",fontsize=9)
for xv,lab in [(0.9,r"$\ell_1$"),(0.5+0.4/3,r"$\ell_3$")]:
    ax[1].axvline(xv,color="#c3c2b7",lw=0.7,ls="--"); ax[1].text(xv,ax[1].get_ylim()[1]*0.6,lab,ha="center",fontsize=8,color="#52514e")
plt.tight_layout(); plt.savefig("../figs/fig_theta_scan.pdf"); plt.savefig("../figs/fig_theta_scan.png")
