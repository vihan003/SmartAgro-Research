from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "SmartAgro_Figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 12,
    "axes.labelsize": 9, "figure.dpi": 130, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY = "#18324B"; BLUE = "#3178A8"; TEAL = "#2A9D8F"; AMBER = "#E9A23B"; RED = "#C65353"; GRAY = "#687782"

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def architecture():
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    def box(x,y,w,h,title,body,color):
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=color,edgecolor="white",lw=1.5,alpha=.96))
        ax.text(x+w/2,y+h-.35,title,ha="center",va="top",color="white",weight="bold",fontsize=10)
        ax.text(x+w/2,y+h-.82,body,ha="center",va="top",color="white",fontsize=8.3,linespacing=1.35)
    box(.2,3.4,3.25,2.05,"A  Crop disease","Leaf image → 16-class classifier\nMobileNetV2 and small CNN\nOffline benchmark; provisional",BLUE)
    box(.2,.55,3.25,2.05,"B  Environmental monitoring","Temperature, humidity, moisture, light\nRecorded datasets + threshold rules\nNo project field sensor deployment",GRAY)
    box(7.55,2.0,3.25,2.05,"C  Regional hotspot risk","NASA FIRMS detections + NASA POWER\nNext-day grid-cell detection score\nHistorical proxy; not field fire risk",TEAL)
    box(4.0,2.0,2.75,2.05,"Decision-support layer","Time-stamped evidence\nQuality and uncertainty checks\nHuman review before action",NAVY)
    for xy1,xy2 in [((3.45,4.35),(4,3.55)),((3.45,1.55),(4,2.45)),((7.55,3.05),(6.75,3.05))]:
        ax.annotate("",xy=xy2,xytext=xy1,arrowprops=dict(arrowstyle="->",lw=1.8,color=GRAY))
    ax.text(5.5,.65,"Modules were evaluated separately; end-to-end integration and field validation remain future work.",ha="center",color=NAVY,weight="bold",fontsize=9)
    save(fig,"fig01_architecture")

def workflow():
    fig, ax = plt.subplots(figsize=(10.8,3.4)); ax.axis("off"); ax.set_xlim(0,10.8); ax.set_ylim(0,3.4)
    nodes=[("Acquire","image / sensor /\nsatellite + weather",BLUE),("Check","schema, timestamps,\nmissingness, domain",GRAY),("Estimate","module-specific\nclass / rule / score",TEAL),("Calibrate","validation data,\nthreshold & uncertainty",AMBER),("Review","human confirmation;\nrecord decision",NAVY)]
    xs=[.15,2.3,4.45,6.6,8.75]
    for i,(head,body,col) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((xs[i],1.05),1.75,1.5,boxstyle="round,pad=.04,rounding_size=.08",fc=col,ec="none"))
        ax.text(xs[i]+.875,2.15,head,ha="center",color="white",weight="bold",fontsize=10)
        ax.text(xs[i]+.875,1.58,body,ha="center",va="center",color="white",fontsize=8)
        if i<4: ax.annotate("",xy=(xs[i+1]-.1,1.8),xytext=(xs[i]+1.8,1.8),arrowprops=dict(arrowstyle="->",lw=1.5,color=GRAY))
    ax.text(5.4,.43,"Outputs retain module-specific provenance; no shared score is issued until joint validation is completed.",ha="center",fontsize=9,color=NAVY)
    save(fig,"fig02_evidence_workflow")

def disease():
    d1=pd.read_csv(HERE/"per_class_grouped__mobilenetv2.csv", index_col=0); d2=pd.read_csv(HERE/"per_class_grouped__small_cnn.csv", index_col=0)
    vals=[("MobileNetV2",.9761,.9692,BLUE),("Small CNN",.7575,.6535,AMBER)]
    fig,ax=plt.subplots(figsize=(7.4,3.8)); x=np.arange(2); w=.34
    ax.bar(x-w/2,[v[1] for v in vals],w,label="Accuracy",color=BLUE)
    ax.bar(x+w/2,[v[2] for v in vals],w,label="Macro-F1",color=TEAL)
    ax.set_xticks(x,[v[0] for v in vals]); ax.set_ylim(0,1.08); ax.set_ylabel("Reported score")
    ax.set_title("Grouped-split disease benchmark (reported aggregates)"); ax.legend(frameon=False,ncol=2,loc="lower right")
    for i,v in enumerate(vals):
        ax.text(i-w/2,v[1]+.02,f"{v[1]:.1%}",ha="center",fontsize=8); ax.text(i+w/2,v[2]+.02,f"{v[2]:.1%}",ha="center",fontsize=8)
    ax.text(.01,-.25,"CSV summary only; run manifest, checkpoints, and per-image predictions were unavailable for independent replay.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig03_disease_aggregate")
    # Class-wise F1 comparison
    a=d1.reset_index().rename(columns={"index":"class"}); b=d2.reset_index().rename(columns={"index":"class"})
    if "class" not in a: a=a.rename(columns={a.columns[0]:"class"})
    if "class" not in b: b=b.rename(columns={b.columns[0]:"class"})
    m=a[["class","f1-score"]].merge(b[["class","f1-score"]],on="class",suffixes=("_mobile","_cnn"))
    m["label"]=m["class"].str.replace("___"," / ",regex=False).str.replace("_"," ",regex=False)
    m=m.sort_values("f1-score_mobile")
    fig,ax=plt.subplots(figsize=(9.2,7.0)); y=np.arange(len(m)); h=.37
    ax.barh(y-h/2,m["f1-score_mobile"],h,color=BLUE,label="MobileNetV2")
    ax.barh(y+h/2,m["f1-score_cnn"],h,color=AMBER,label="Small CNN")
    ax.set_yticks(y,m.label,fontsize=7); ax.set_xlim(0,1.08); ax.set_xlabel("Per-class F1 (reported)")
    ax.set_title("Per-class F1 across the 16 evaluated labels"); ax.legend(frameon=False,loc="lower right")
    ax.grid(axis="x",alpha=.2); save(fig,"fig04_disease_per_class")

def environmental():
    s=pd.read_csv(HERE/"sensor_statistics.csv"); s=s[s.dataset_key.eq("smartagro_iot")]
    wanted=[("temperature_c","Air temperature (°C)"),("humidity_pct","Relative humidity (%)"),("soil_moisture_pct","Soil-moisture source scale (%)"),("soil_ph","Soil pH"),("soil_temperature_c","Soil temperature (°C)")]
    fig,axs=plt.subplots(1,5,figsize=(11,3.2))
    for ax,(key,title) in zip(axs,wanted):
        r=s[s.variable.eq(key)]
        if not len(r): ax.axis("off"); continue
        r=r.iloc[0]; ax.vlines(0,r["min"],r["max"],color=BLUE,lw=6,alpha=.8); ax.scatter([0],[r["median"]],s=50,color=AMBER,zorder=3)
        ax.set_xlim(-.6,.6); ax.set_xticks([]); ax.set_title(title,fontsize=8); ax.set_ylabel("Observed range")
        ax.text(.05,.97,f"min {r['min']:.2f}\nmedian {r['median']:.2f}\nmax {r['max']:.2f}\nn={int(r['count'])}",transform=ax.transAxes,va="top",fontsize=7)
    fig.suptitle("Recorded SmartAgroTech archive: 101 matched readings",y=1.04,fontsize=12)
    fig.text(.5,-.03,"Humidity and pH are constant in this short archive; one very low moisture value is retained.",ha="center",fontsize=8,color=GRAY)
    save(fig,"fig05_recorded_sensor_ranges")
    cats=pd.read_csv(HERE/"stress_category_summary.csv"); cats=cats[cats.dataset_key.eq("smartagro_iot")]
    order=["Normal","Low stress","Moderate stress","High stress"]
    vals={r.category:int(r.count) for r in cats.itertuples()}
    fig,ax=plt.subplots(figsize=(6.7,3.6)); colors=[GRAY,BLUE,AMBER,RED]
    ax.bar(order,[vals.get(x,0) for x in order],color=colors)
    ax.set_ylabel("Readings"); ax.set_title("Rule-generated categories for the recorded archive")
    for i,x in enumerate(order): ax.text(i,vals.get(x,0)+1,str(vals.get(x,0)),ha="center")
    ax.text(.02,-.22,"Preset thresholds only; these are not field-diagnosed crop stress labels.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig06_rule_category_counts")
    ev=pd.read_csv(HERE/"evaluation_summary.csv"); ev=ev[ev.dataset.eq("predictive_irrigation")].iloc[0]
    fig,ax=plt.subplots(figsize=(6.7,3.4)); ax.bar(["Test pump-off","Test pump-on"],[ev.test_class_0,ev.test_class_1],color=[BLUE,AMBER])
    ax.set_ylabel("Test observations"); ax.set_title("Predictive-irrigation test set class support")
    for i,v in enumerate([ev.test_class_0,ev.test_class_1]): ax.text(i,v+2,str(int(v)),ha="center")
    ax.text(.02,-.24,"Only 2 positive examples; perfect classifier scores are not evidence of reliable generalization.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig07_pump_test_support")

def fire():
    m=pd.read_csv(HERE/"fire_oct_nov_counts.csv"); m["month_label"]=m.month.map({10:"Oct",11:"Nov"})
    pv=m.pivot(index="year",columns="month_label",values="hotspot_records").fillna(0)
    fig,ax=plt.subplots(figsize=(8,3.8)); pv.reindex(columns=["Oct","Nov"],fill_value=0).plot(kind="bar",ax=ax,color=[BLUE,TEAL],width=.78)
    ax.set_title("VIIRS/S-NPP hotspot records by year (October–November)"); ax.set_ylabel("Detection records"); ax.set_xlabel("Year")
    ax.legend(title="Month",frameon=False); ax.tick_params(axis="x",rotation=0); ax.grid(axis="y",alpha=.2)
    ax.text(.01,-.23,"Records are satellite thermal detections; they do not identify cause, crop residue burning, or confirmed fire events.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig08_fire_oct_nov_counts")
    c=pd.read_csv(HERE/"fire_grid_cell_rates.csv")
    fig,ax=plt.subplots(figsize=(7,5.1)); sc=ax.scatter(c.lon,c.lat,c=c.next_day_detection_rate,s=35+80*c.cell_days/c.cell_days.max(),cmap="YlOrRd",vmin=0,vmax=1,edgecolor="white",linewidth=.45)
    cb=fig.colorbar(sc,ax=ax); cb.set_label("Share of included days followed by a cell detection")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.set_title("Grid-cell next-day detection prevalence")
    ax.grid(alpha=.2); ax.text(.01,-.17,"Circle size indicates included cell-days. No cropland mask or spatial holdout was used.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig09_fire_grid_rates")
    p=pd.read_csv(HERE/"fire_primary_results.csv"); p=p[p.model.isin(["persistence (fire yesterday)","climatology (day-of-season rate)","weather index (hot, dry, windy)","logistic regression","gradient-boosted trees"])]
    fig,ax=plt.subplots(figsize=(8,4.4)); y=np.arange(len(p)); scores=p.average_precision.to_numpy(); lo=scores-p.ci95_low.fillna(p.average_precision).to_numpy(); hi=p.ci95_high.fillna(p.average_precision).to_numpy()-scores
    color=[GRAY,GRAY,GRAY,BLUE,TEAL]
    ax.errorbar(scores,y,xerr=[lo,hi],fmt="none",ecolor=GRAY,capsize=3,lw=1.2)
    ax.scatter(scores,y,c=color,s=65,zorder=3); ax.axvline(.4954463,color=RED,ls="--",lw=1,label="Test prevalence (0.495)")
    ax.set_yticks(y,p.model); ax.invert_yaxis(); ax.set_xlim(.35,.98); ax.set_xlabel("Average precision (AUPRC)")
    ax.set_title("Primary fire benchmark with 95% weekly block-bootstrap intervals")
    ax.legend(frameon=False,loc="lower right"); ax.grid(axis="x",alpha=.2)
    save(fig,"fig10_fire_primary_auprc")
    aux=pd.read_csv(HERE/"fire_auxiliary_results.csv"); aux=aux[aux.model_display.isin(["Logistic regression","Histogram gradient boosting","Random forest","XGBoost"])].copy()
    aux=aux.sort_values("AUPRC")
    fig,ax=plt.subplots(figsize=(7.4,3.6)); ax.barh(aux.model_display,aux.AUPRC,color=[GRAY,BLUE,AMBER,TEAL])
    ax.set_xlim(.75,.92); ax.set_xlabel("AUPRC at fixed 0.5 threshold protocol"); ax.set_title("Auxiliary model comparison (descriptive; no intervals)")
    for i,v in enumerate(aux.AUPRC): ax.text(v+.002,i,f"{v:.3f}",va="center",fontsize=8)
    ax.text(.01,-.24,"These auxiliary values have no bootstrap intervals and use a different operating-threshold protocol.",transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,"fig11_fire_auxiliary_models")
    pr=pd.read_csv(HERE/"fire_gbt_pr_curve.csv")
    fig,ax=plt.subplots(figsize=(5.5,4.6)); ax.plot(pr.recall,pr.precision,color=TEAL,lw=2,label="Histogram gradient boosting")
    ax.axhline(.4954463,color=RED,ls="--",lw=1,label="Test prevalence")
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="Primary GBT precision–recall curve")
    ax.grid(alpha=.2); ax.legend(frameon=False,loc="lower left")
    save(fig,"fig12_fire_pr_curve")

def main():
    architecture(); workflow(); disease(); environmental(); fire()
    print(f"Wrote {len(list(OUT.glob('*.png')))} PNG and {len(list(OUT.glob('*.pdf')))} PDF figures to {OUT}")

if __name__ == "__main__": main()
