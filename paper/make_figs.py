import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
K="#1f4e79"
# Fig: leakage, 4 tasks x 3 gains
tasks=["CD","MI","STTC","HYP"]
G={"$\\Delta_{\\mathrm{rand}}$ (vs. untrained network)":[(.028,.020,.038),(.017,.010,.024),(.013,.008,.017),(.016,.008,.026)],
   "$\\Delta_{\\mathrm{raw}}$ (vs. raw median beat)":[(.044,.029,.058),(.022,.012,.033),(.003,-.003,.008),(.020,.007,.033)],
   "$\\Delta_{\\mathrm{demo}}$ (beyond age and sex)":[(.130,.113,.146),(.122,.108,.137),(.190,.174,.207),(.174,.154,.195)]}
fig,ax=plt.subplots(1,3,figsize=(7.16,1.9),sharey=True)
y=np.arange(4)[::-1]
for a,(t,v) in zip(ax,G.items()):
    e=np.array(v); a.errorbar(e[:,0],y,xerr=[e[:,0]-e[:,1],e[:,2]-e[:,0]],fmt="o",ms=3.5,capsize=2.5,color=K,lw=1)
    a.axvline(0,ls="--",lw=.7,color="grey"); a.set_title(t,fontsize=8); a.set_xlabel("AUROC gain (95% CI)")
ax[0].set_yticks(y); ax[0].set_yticklabels([f"{t} vs. NORM" for t in tasks])
plt.tight_layout(); fig.savefig("figures/fig_leakage.pdf",bbox_inches="tight")
# Fig: coupling + persistence
ep=[1,2,3,4,5,6,8,10,12,15,19]; eer=[.189,.117,.096,.082,.078,.067,.070,.058,.064,.068,.064]; dr=[.015,.016,.010,.014,.018,.021,.023,.023,.025,.022,.021]
fig,ax=plt.subplots(1,2,figsize=(7.16,2.1))
ax[0].plot(eer,dr,"-o",ms=3,color=K,lw=1)
for e,x,d in zip(ep,eer,dr): ax[0].annotate(str(e),(x,d),xytext=(3,2),textcoords="offset points",fontsize=6)
ax[0].invert_xaxis(); ax[0].set_xlabel("Identity EER on ID-eval (lower is better)"); ax[0].set_ylabel("$\\Delta_{\\mathrm{rand}}$ (probe-dev, CD)")
ax[0].set_title("(a) Utility--leakage coupling, $\\rho=-0.75$",fontsize=8)
P={"MI":(.766,.017,.010,.024),"CD":(.723,.028,.020,.038),"STTC":(.671,.013,.008,.017),"HYP":(.615,.016,.008,.026)}
for k,(p,e,l,h) in P.items():
    ax[1].errorbar(p,e,yerr=[[e-l],[h-e]],fmt="o",ms=3.5,capsize=2.5,color=K,lw=1); ax[1].annotate(k,(p,e),xytext=(4,2),textcoords="offset points",fontsize=7)
ax[1].axhline(0,ls="--",lw=.7,color="grey"); ax[1].set_xlabel("Label persistence to next visit"); ax[1].set_ylabel("$\\Delta_{\\mathrm{rand}}$ (lockbox)")
ax[1].set_title("(b) Persistence vs. leakage (4 classes, descriptive)",fontsize=8)
plt.tight_layout(); fig.savefig("figures/fig_coupling.pdf",bbox_inches="tight")
# Fig: robustness
R=[("PTB-XL lead I, seed 0 (primary)",.028,.020,.038),("PTB-XL lead I, seed 1",.030,.021,.039),("PTB-XL lead I, seed 2",.019,.010,.029),
   ("PTB-XL lead II",.008,.002,.014),("CODE-15%, frozen model",.021,.007,.038),("Attacker: MLP",.012,.003,.023),("Attacker: $k$NN",.046,.034,.058),
   ("Biometric: Siamese contrastive",.017,.010,.025)]
fig,a=plt.subplots(figsize=(3.5,2.3)); y=np.arange(len(R))[::-1]; e=np.array([r[1:] for r in R])
a.errorbar(e[:,0],y,xerr=[e[:,0]-e[:,1],e[:,2]-e[:,0]],fmt="o",ms=3.5,capsize=2.5,color=K,lw=1)
a.axvline(0,ls="--",lw=.7,color="grey"); a.set_yticks(y); a.set_yticklabels([r[0] for r in R],fontsize=7); a.set_xlabel("$\\Delta_{\\mathrm{rand}}$, CD (95% CI)")
plt.tight_layout(); fig.savefig("figures/fig_robustness.pdf",bbox_inches="tight")
# Fig: attribution overlap differences
D=[("vs. shuffled-label diagnosis model",.077,.070,.084),("vs. untrained identity encoder",.070,.065,.074)]
fig,a=plt.subplots(figsize=(3.5,1.2)); y=np.arange(2)[::-1]; e=np.array([d[1:] for d in D])
a.errorbar(e[:,0],y,xerr=[e[:,0]-e[:,1],e[:,2]-e[:,0]],fmt="o",ms=3.5,capsize=2.5,color=K,lw=1)
a.axvline(0,ls="--",lw=.7,color="grey"); a.set_yticks(y); a.set_yticklabels([d[0] for d in D],fontsize=7); a.set_xlim(-0.01,0.1)
a.set_xlabel("Excess attribution overlap (95% CI)")
plt.tight_layout(); fig.savefig("figures/fig_attribution.pdf",bbox_inches="tight")
print("ok")
