"""Figure: gene LENGTH is the granularity dial.
(a) broader (more scattered) PACS cluster -> shorter raw-mean centroid (r=-0.985).
(b) scaling a single cluster's gene magnitude down walks its decoded label from a
    specific subfield up to the base-model prior (abstraction ladder)."""
import sys, collections
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
DISK = ROOT / "data/aps/embeddings/qwen_norm_lora_emb.npz"
OUT = ROOT / "exps/2026-06-11-baseline-trees"
SAMPLE = 6000
rng = np.random.default_rng(0)
NODES = [("root","root",None),("3","main",3),("75","division","75"),("75.10","sub","75.10"),
("75.30","sub","75.30"),("71","division","71"),("71.10","sub","71.10"),("71.20","sub","71.20"),
("0","main",0),("03","division","03"),("03.67","sub","03.67"),("03.65","sub","03.65"),
("05","division","05"),("05.45","sub","05.45"),("05.40","sub","05.40"),("6","main",6),
("11","division","11"),("11.15","sub","11.15"),("11.10","sub","11.10"),("12","division","12"),
("12.38","sub","12.38"),("12.60","sub","12.60"),("2","main",2),("42","division","42"),
("42.50","sub","42.50"),("42.65","sub","42.65"),("47","division","47"),("47.27","sub","47.27"),
("47.20","sub","47.20"),("1","main",1),("4","main",4),("5","main",5),("7","main",7),("8","main",8)]
LV = {"root":0,"main":1,"division":2,"sub":3}
LNAME = ["root","field","division","sub-subfield"]
COL = ["#b2182b","#ef8a62","#67a9cf","#2166ac"]

dz = np.load(DISK); genes = dz["embeddings"]
pid2row = {int(p):i for i,p in enumerate(dz["paper_ids"])}
pg = pd.read_parquet(CA/"results/paper_groups.parquet").dropna(subset=["main_class_id","division","subdivision"]).copy()
pg["paper_id"]=pg.paper_id.astype(int); pg["main_class_id"]=pg.main_class_id.astype(int)
pg["division"]=pg.division.astype(str); pg["subdivision"]=pg.subdivision.astype(str)
def rows_of(level,val):
    if level=="root": ids=pg.paper_id.values
    elif level=="main": ids=pg[pg.main_class_id==val].paper_id.values
    elif level=="division": ids=pg[pg.division==val].paper_id.values
    else: ids=pg[pg.subdivision==val].paper_id.values
    return np.array([pid2row[int(p)] for p in ids if int(p) in pid2row])

scat=[]; cnorm=[]; lev=[]; recs=[]
for key,level,val in NODES:
    r=rows_of(level,val)
    if len(r)==0: continue
    smp = r if len(r)<=SAMPLE else rng.choice(r,SAMPLE,replace=False)
    V=genes[smp].astype(np.float32)
    cent=V.mean(0); cn=float(np.linalg.norm(cent))
    u=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); cu=cent/(cn+1e-9)
    sc=float(1-(u@cu).mean())
    scat.append(sc); cnorm.append(cn); lev.append(LV[level])
    recs.append((key,LV[level],len(r),cn,sc))
scat=np.array(scat); cnorm=np.array(cnorm); lev=np.array(lev)
r_pear=np.corrcoef(cnorm,scat)[0,1]
pd.DataFrame(recs,columns=["node","level","n","centroid_norm","scatter"]).to_csv(OUT/"length_vs_breadth.csv",index=False)

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "font.family":"DejaVu Sans"})
fig,(ax,axb)=plt.subplots(1,2,figsize=(9.2,3.5),gridspec_kw={"width_ratios":[1,1.15]})

# panel (a)
for l in range(4):
    m=lev==l
    ax.scatter(scat[m],cnorm[m],s=46,c=COL[l],edgecolor="white",linewidth=0.6,
               label=LNAME[l],zorder=3)
b,a=np.polyfit(scat,cnorm,1); xs=np.linspace(scat.min(),scat.max(),50)
ax.plot(xs,a+b*xs,"--",color="0.35",lw=1.2,zorder=2)
# annotate root
ri=[i for i,rr in enumerate(recs) if rr[0]=="root"][0]
ax.annotate("whole corpus\n(root)",(scat[ri],cnorm[ri]),xytext=(scat[ri]-0.002,cnorm[ri]-0.06),
            fontsize=8.5,ha="left",color=COL[0])
ax.set_xlabel("cluster scatter  (1 $-$ mean cos to centroid)")
ax.set_ylabel("centroid length  $\\|\\bar{\\mathcal{V}}\\|$")
ax.text(0.04,0.06,f"$r={r_pear:.2f}$",transform=ax.transAxes,fontsize=11,fontweight="bold")
ax.legend(frameon=False,fontsize=8.2,loc="upper right",handletextpad=0.2,borderpad=0.2)
ax.set_title("(a) broader cluster $\\rightarrow$ shorter gene",fontsize=10.5)

# panel (b): abstraction ladder from magnitude_trajectory.py results
scales=[2.0,1.0,0.7,0.5,0.3,0.1,0.0]
ladder={
 "Condensed matter":["Superconducting nanoscale","Superconductivity in 2D","Quantum phase transitions","Quantum optics","Quantum computing","Genetic engineering","Genetic engineering"],
 "Nuclear physics":["Nuclear reaction dynamics","Nuclear reaction dynamics","Nuclear structure & reactions","Quantum mechanics","Quantum computing","Genetic engineering","Genetic engineering"],
 "Atomic physics":["Atomic collision dynamics","Atomic physics","Quantum scattering theory","Quantum optics","Quantum computing","Genetic engineering","Genetic engineering"],
}
axb.axis("off")
y0=0.84; dy=0.118
axb.text(0.0,1.05,"(b) shrinking gene length $\\rightarrow$ more general label",
         fontsize=10.5,transform=axb.transAxes)
xs_cluster={"Condensed matter":0.30,"Nuclear physics":0.30,"Atomic physics":0.30}
# draw one cluster (condensed matter) as the ladder, with scale axis
cl="Condensed matter"
labs=ladder[cl]
axb.text(0.13,y0+dy*0.62,"scale",fontsize=8.5,ha="right",style="italic",transform=axb.transAxes)
axb.text(0.18,y0+dy*0.62,f"decoded label  ({cl} cluster)",fontsize=8.5,ha="left",style="italic",transform=axb.transAxes)
for i,(s,lab) in enumerate(zip(scales,labs)):
    y=y0-i*dy
    shade=str(0.15+0.62*(i/(len(scales)-1)))
    axb.text(0.13,y,f"{s:g}",fontsize=9,ha="right",transform=axb.transAxes,color="0.2")
    axb.text(0.18,y,lab,fontsize=9,ha="left",transform=axb.transAxes,color=shade)
# arrow for increasing generality
axb.annotate("",xy=(0.045,y0-(len(scales)-1)*dy),xytext=(0.045,y0),
             xycoords="axes fraction",arrowprops=dict(arrowstyle="->",color="0.4",lw=1.3))
axb.text(0.012,y0-(len(scales)-1)*dy*0.5,"more general",rotation=90,va="center",
         fontsize=8.5,color="0.4",transform=axb.transAxes)
axb.text(0.18,y0-len(scales)*dy+0.01,"(scale 0 = no gene: base-model prior)",
         fontsize=7.8,style="italic",color="0.55",transform=axb.transAxes,ha="left")

plt.tight_layout(w_pad=2.0)
for ext in ("pdf","png"):
    fig.savefig(OUT/f"length_dial.{ext}",dpi=200,bbox_inches="tight")
print("r=",r_pear)
print("wrote", OUT/"length_dial.pdf")
