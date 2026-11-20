import numpy as np
from fig_common import *
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),gridspec_kw=dict(width_ratios=[1.25,1,1]))
d=np.load("../data/fine_full10_fv.npz")
ax[0].semilogy(d["l"],d["Cinf"],color=S1,lw=1.0,label=r"$\overline{C}_\infty$")
ax[0].semilogy(d["l"],d["C"][:,d["ts"]<=1000].mean(1),"--",color=S2,lw=1.0,label=r"$\overline{C}_{[0,1000]}$")
for n,sh in [(1,0.9),(3,0.6398),(5,0.5836)]:
    ax[0].axvline(0.5+0.4/n,color="#c3c2b7",lw=0.7,ls="--")
    ax[0].text(0.5+0.4/n,0.33,rf"$\ell_{n}$",ha="center",fontsize=7.5,color="#52514e")
ax[0].set_ylim(0.25,60); ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel("complexity"); ax[0].legend(frameon=False,fontsize=7,loc="upper left",bbox_to_anchor=(0.0,0.9))
ax[0].text(0.97,0.93,"(a) $N=10$",transform=ax[0].transAxes,ha="right",fontsize=8)
d3=np.load("../data/fine_res3_12.npz"); d5=np.load("../data/fine_res5_10.npz")
ax[1].plot((d3["l"]-0.6398)*1e3,d3["Cinf"]/d3["Cinf"].max(),"o-",ms=2.5,color=S3,lw=1,label=r"$n=3$, $N=12$")
ax[1].plot((d5["l"]-0.5836)*1e3,d5["Cinf"]/d5["Cinf"].max(),"s-",ms=2.0,color=S4,lw=1,label=r"$n=5$, $N=10$")
ax[1].axvline(0,color="#8a8984",lw=0.8,ls=":")
ax[1].set_xlabel(r"$(\ell-\ell_n^{(2)})\times10^{3}$"); ax[1].set_ylabel(r"$\overline{C}_\infty/\max$"); ax[1].set_xlim(-8,8)
ax[1].legend(frameon=False,fontsize=6.5,loc="center left",bbox_to_anchor=(0.0,0.62)); ax[1].text(0.97,0.05,"(b)",transform=ax[1].transAxes,ha="right",fontsize=8)
for tag,N,c,ls in [("scan_full10",10,S2,"-"),("scan_full12",12,S3,"--")]:
    s=np.load(f"../data/{tag}.npz"); k=np.argmin(abs(s["l"]-0.9)); m=s["ts"]<=130
    ax[2].plot(s["ts"][m],s["C"][k][m],ls,color=c,lw=1.0,label=f"$N={N}$")
for j in range(1,5): ax[2].axvline(j*np.pi/0.1,color="#c3c2b7",lw=0.7,ls=":")
ax[2].set_xlabel(r"$t$"); ax[2].set_ylabel(r"$C(t)$ at $\ell=\ell_1$"); ax[2].legend(frameon=False,fontsize=7,loc="upper right",bbox_to_anchor=(1.0,0.88))
ax[2].text(0.03,0.93,r"(c) ticks: $t=k\pi/x$",transform=ax[2].transAxes,fontsize=7.5)
for a in ax: a.grid(color=GRID,lw=0.5)
plt.tight_layout(); plt.savefig("../figs/fig_resonances.pdf"); plt.savefig("../figs/fig_resonances.png",dpi=150)
