from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_results" / "fire_risk"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import exp_c_fire_risk as fire

CACHE = ROOT / "outputs" / "results_c" / "cache"
YEARS = list(range(2017, 2024))

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
                     "axes.labelsize":9,"figure.facecolor":"white","axes.facecolor":"white"})

def savefig(fig, name):
    fig.savefig(FIG/f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# Load only real cache contents.
firms_frames=[]
for p in sorted(CACHE.glob("firms_*.csv")):
    d=pd.read_csv(p)
    if "confidence" in d:
        d=d[d["confidence"].astype(str).str.lower()!= "l"]
    firms_frames.append(d)
fires=pd.concat(firms_frames, ignore_index=True).drop_duplicates()

power=[]
for p in sorted(CACHE.glob("power_*.json")):
    pjson=json.loads(p.read_text())
    pa=pjson["properties"]["parameter"]
    d=pd.DataFrame(pa)
    d.index=pd.to_datetime(d.index,format="%Y%m%d")
    d=d.replace(-999,np.nan).reset_index().rename(columns={"index":"date"})
    lat,lon=map(float,p.stem.replace("power_","").split("_"))
    d["lat"]=lat; d["lon"]=lon
    d["cell"]=fire.cell_id(np.array([lat]),np.array([lon]))[0]
    power.append(d)
wx=pd.concat(power,ignore_index=True)
df=fire.build(fires,wx)

# Reproduce the existing exact split and evaluate.
info,res=fire.evaluate(df,2020,2021,str(OUT))

# Satellite hotspot description.
fires["acq_date"]=pd.to_datetime(fires["acq_date"])
fires["year"]=fires["acq_date"].dt.year
by_month=fires.groupby(fires["acq_date"].dt.to_period("M")).size()
monthly=fires.groupby(fires["acq_date"].dt.month).size()

# Dataset summary.
dataset_summary=pd.DataFrame([{
    "dataset":"NASA FIRMS VIIRS_SNPP active-fire hotspots",
    "source":"https://firms.modaps.eosdis.nasa.gov/",
    "bbox":"74.0E,28.4N to 77.6E,32.6N",
    "temporal_range":f"{fires.acq_date.min().date()} to {fires.acq_date.max().date()}",
    "spatial_resolution":"Point detections / 0.5° x 0.625° grid labels",
    "temporal_resolution":"5-day API windows; daily detections",
    "records":len(fires),"cache_files":len(firms_frames),"unique_dates":fires.acq_date.nunique(),
    "missing_value_percent":float(fires.isna().mean().mean()*100),"target":"None; hotspot observations",
    "limitations":"Satellite-detected thermal anomalies, not confirmed agricultural fires"
},{
    "dataset":"NASA POWER daily meteorology",
    "source":"https://power.larc.nasa.gov/",
    "bbox":"74.0E,28.4N to 77.6E,32.6N",
    "temporal_range":"2017-01-01 to 2023-12-31",
    "spatial_resolution":"54 grid-point locations at 0.5° x 0.625°",
    "temporal_resolution":"Daily",
    "records":len(wx),"cache_files":len(power),"unique_dates":wx.date.nunique(),
    "missing_value_percent":float(wx.isna().mean().mean()*100),"target":"None",
    "limitations":"Point meteorology used as observed-weather proxy, not forecast"
},{
    "dataset":"Reconstructed cell-day fire-risk dataset",
    "source":"FIRMS detections joined to NASA POWER",
    "bbox":"74.0E,28.4N to 77.6E,32.6N",
    "temporal_range":f"{df.date.min().date()} to {df.date.max().date()}",
    "spatial_resolution":"54 grid cells",
    "temporal_resolution":"Daily, October-November labels",
    "records":len(df),"cache_files":"91 FIRMS + 54 POWER","unique_dates":df.date.nunique(),
    "missing_value_percent":float(df.isna().mean().mean()*100),"target":"VIIRS detection on next day",
    "limitations":"Regional detection-risk label; no confirmed fire/crop mask"
}])
dataset_summary.to_csv(OUT/"dataset_summary.csv",index=False)

# Evaluation summary exact existing and additional models.
tr=df[df.year<=2020]; va=df[df.year==2021]; te=df[df.year>2021]
models={}
models["logistic regression"]={"fit":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(tr[fire.FEATS].fillna(0),tr.y)}
models["histogram gradient boosting"]={"fit":HistGradientBoostingClassifier(max_depth=3,learning_rate=.08,max_iter=150,random_state=0).fit(tr[fire.FEATS],tr.y)}
models["random forest"]={"fit":RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=10,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[fire.FEATS].fillna(0),tr.y)}
if XGBClassifier:
    models["xgboost"]={"fit":XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=0,n_jobs=-1).fit(tr[fire.FEATS].fillna(0),tr.y)}

for k in ("logistic regression","histogram gradient boosting","random forest","xgboost"):
    if k not in models: continue
    m=models[k]["fit"]
    prob=m.predict_proba(te[fire.FEATS].fillna(0))[:,1]
    pred=prob>=0.5
    ap=average_precision_score(te.y,prob)
    roc=roc_auc_score(te.y,prob)
    brier=brier_score_loss(te.y,prob)
    res[f"{k} (0.5 threshold)"]={"AUPRC":float(ap),"ROC_AUC":float(roc),"precision":float(precision_score(te.y,pred,zero_division=0)),"recall":float(recall_score(te.y,pred,zero_division=0)),"f1":float(f1_score(te.y,pred,zero_division=0)),"brier":float(brier),"threshold":0.5}

# Model comparison for existing baselines + exact reference models.
comparison=[]
reference=res
for name,m in res.items():
    comparison.append({"model":name,"AUPRC":m.get("AUPRC",np.nan),"ROC_AUC":m.get("ROC_AUC",np.nan),"precision":m.get("precision",np.nan),"recall":m.get("recall",np.nan),"f1":m.get("f1",np.nan),"Brier":m.get("brier",np.nan),"missed_event_rate":m.get("missed_event_rate",np.nan),"false_alarms_per_100_negative_cell_days":m.get("false_alarms_per_100_negative_cell_days",np.nan),"alert_share_pct":m.get("alert_share_pct",np.nan)})
pd.DataFrame(comparison).to_csv(OUT/"model_comparison.csv",index=False)

# Add full evaluation details.
eval_details=pd.DataFrame([{
    "split":"train","years":"<=2020","rows":len(tr),"positives":int(tr.y.sum()),"negatives":int((tr.y==0).sum()),"base_rate":float(tr.y.mean()),"method":"Calendar-year chronological"
},{
    "split":"validation","years":"2021","rows":len(va),"positives":int(va.y.sum()),"negatives":int((va.y==0).sum()),"base_rate":float(va.y.mean()),"method":"Calendar-year chronological"
},{
    "split":"test","years":"2022-2023","rows":len(te),"positives":int(te.y.sum()),"negatives":int((te.y==0).sum()),"base_rate":float(te.y.mean()),"method":"Calendar-year chronological"
}])
eval_details["target"]= "at least one non-low-confidence VIIRS detection in cell on next day"
eval_details.to_csv(OUT/"evaluation_details.csv",index=False)

# Geographic hotspot map scatter.
fig,ax=plt.subplots(figsize=(8,6.5))
h=fires.dropna(subset=["latitude","longitude"])
ax.scatter(h.longitude,h.latitude,s=np.clip(h.get("frp",pd.Series(1,index=h.index)).fillna(1).astype(float)*.25,.2,30),alpha=.25,c="#B65252",label="VIIRS hotspot")
ax.set_xlim(fire.BBOX[0],fire.BBOX[2]);ax.set_ylim(fire.BBOX[1],fire.BBOX[3])
ax.set_title("NASA FIRMS VIIRS hotspot observations, 2017–2023")
ax.set_xlabel("Longitude (°E)");ax.set_ylabel("Latitude (°N)");ax.grid(alpha=.25);ax.legend(frameon=False)
savefig(fig,"fig01_hotspot_geographic_distribution")

# Temporal distribution.
fig,ax=plt.subplots(figsize=(10,4.8))
by_month.plot(ax=ax,color="#3977A8",lw=1)
ax.set_title("VIIRS hotspot observations over time")
ax.set_xlabel("Month");ax.set_ylabel("Hotspot observation count");ax.grid(alpha=.25);savefig(fig,"fig02_hotspot_monthly_time_series")

fig,ax=plt.subplots(figsize=(8,5))
monthly.plot(kind="bar",ax=ax,color="#27877B")
ax.set_title("Seasonal hotspot observations (October–November subset)")
ax.set_xlabel("Month");ax.set_ylabel("Hotspot observation count");ax.set_xticklabels(["Oct","Nov"] if len(monthly)==2 else monthly.index);ax.grid(axis="y",alpha=.25)
savefig(fig,"fig03_seasonal_hotspot_distribution")

# Weather distributions.
season=wx[wx.date.dt.month.isin(fire.SEASON)]
fig,axes=plt.subplots(2,2,figsize=(10,7.5))
season.T2M.plot.hist(bins=30,ax=axes[0,0],color="#D56D42",edgecolor="white")
axes[0,0].set(title="Temperature",xlabel="Temperature (°C)",ylabel="Cell-days")
season.RH2M.plot.hist(bins=30,ax=axes[0,1],color="#3977A8",edgecolor="white")
axes[0,1].set(title="Relative humidity",xlabel="Relative humidity (%)",ylabel="Cell-days")
season.WS2M.plot.hist(bins=30,ax=axes[1,0],color="#27877B",edgecolor="white")
axes[1,0].set(title="Wind speed",xlabel="Wind speed (m/s)",ylabel="Cell-days")
season.PRECTOTCORR.plot.hist(bins=30,ax=axes[1,1],color="#D89B2B",edgecolor="white")
axes[1,1].set(title="Precipitation",xlabel="Precipitation (mm/day)",ylabel="Cell-days")
fig.suptitle("NASA POWER weather distributions, October–November",fontsize=13);fig.tight_layout();savefig(fig,"fig04_weather_distributions")

# Fire/non-fire class distribution.
counts={"Fire detection next day":int(te.y.sum()),"No detection next day":int((te.y==0).sum())}
fig,ax=plt.subplots(figsize=(7,4.8));ax.bar(counts.keys(),counts.values(),color=["#B65252","#3977A8"])
for i,(k,v) in enumerate(counts.items()) :
    ax.text(i,v+50,f"{v}",ha="center",fontweight="bold")
ax.set_title("Test-set class distribution (2022–2023)");ax.set_ylabel("Cell-days");ax.grid(axis="y",alpha=.25);savefig(fig,"fig05_fire_nonfire_test_distribution")

# Existing model AUPRC comparison.
order=["persistence (fire yesterday)","climatology (day-of-season rate)","weather index (hot, dry, windy)","logistic regression","gradient-boosted trees"]
labels=["Persistence","Climatology","Hot/dry/windy","Logistic regression","Gradient-boosted trees"]
vals=[reference[k]["AUPRC"] for k in order]
ci=[reference[k]["AUPRC_CI95_block_bootstrap"] for k in order]
errors=np.array([[v-c[0] for v,c in zip(vals,ci)],[c[1]-v for v,c in zip(vals,ci)]])
fig,ax=plt.subplots(figsize=(9,5.2));ax.bar(labels,vals,yerr=errors,capsize=4,color=["#687887","#687887","#D89B2B","#3977A8","#27877B"])
for i,v in enumerate(vals):ax.text(i,v+.015,f"{v:.3f}",ha="center",fontweight="bold")
ax.set_ylim(0,1.02);ax.set_ylabel("AUPRC");ax.set_title("Reproduced fire-risk model comparison");ax.tick_params(axis="x",rotation=15);ax.grid(axis="y",alpha=.25);savefig(fig,"fig06_model_comparison_auprc")

# Precision-recall and ROC for selected GBT reference model by reproducing probability.
lrm=models["logistic regression"]["fit"] if "logistic regression" in models else None
gbm=models["histogram gradient boosting"]["fit"]
prob=gbm.predict_proba(te[fire.FEATS])[:,1]
precision,recall,_=precision_recall_curve(te.y,prob);fpr,tpr,_=roc_curve(te.y,prob)
cm=confusion_matrix(te.y,prob>=.5)
fig,ax=plt.subplots(figsize=(7,5));ax.plot(recall,precision,color="#27877B",lw=2);ax.set_xlabel("Recall");ax.set_ylabel("Precision");ax.set_title("Gradient-boosted test precision-recall curve");ax.grid(alpha=.25);savefig(fig,"fig07_precision_recall_curve")
fig,ax=plt.subplots(figsize=(7,5));ax.plot(fpr,tpr,color="#27877B",lw=2);ax.plot([0,1],[0,1],"k--",lw=1);ax.set_xlabel("False positive rate");ax.set_ylabel("True positive rate");ax.set_title("Gradient-boosted test ROC curve");ax.grid(alpha=.25);savefig(fig,"fig08_roc_curve")
fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap="Blues");ax.set_xticks([0,1],["No detection","Detection"]);ax.set_yticks([0,1],["No detection","Detection"])
for i in range(2):
    for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontweight="bold")
ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title("Gradient-boosted 0.5-threshold confusion matrix");savefig(fig,"fig09_gbt_confusion_matrix")

# Feature importance / permutation-free HGB has no native importance. Random forest is stable and documented only as auxiliary.
rf=models["random forest"]["fit"]
from sklearn.inspection import permutation_importance
perm=permutation_importance(rf,tr[fire.FEATS].fillna(0),tr.y,n_repeats=5,random_state=0,scoring="average_precision")
imp=pd.Series(perm.importances_mean,index=fire.FEATS).sort_values()
fig,ax=plt.subplots(figsize=(8,5));imp.plot(kind="barh",ax=ax,color="#3977A8");ax.set_xlabel("Mean validation permutation importance (AUPRC)");ax.set_title("Auxiliary random forest feature permutation importance");ax.grid(axis="x",alpha=.25);savefig(fig,"fig10_feature_permutation_importance")

# Save machine-readable exact outputs.
output={"dataset_setup":dataset_summary.to_dict("records"),"evaluation":info,"existing_reference_models":{k:reference[k] for k in order},"added_0_5_threshold_models":{k:v for k,v in res.items() if k not in order},"auditable_source":"outputs/results_c/cache","target":"At least one non-low-confidence VIIRS detection in grid cell on day t+1","limitations":["Regional 0.5° × 0.625° grid cells, not field-level risk","Observed weather is optimistic; not forecast weather","Hotspots are satellite-detected anomalies, not confirmed agricultural fires","No cropland mask","Spatially adjacent grid cells remain in same calendar-year splits","No independent spatial holdout"]}
(OUT/"fire_risk_analysis.json").write_text(json.dumps(output,indent=2,default=float),encoding="utf-8")

print(json.dumps({"rows":len(df),"hotspots":len(fires),"firms_files":len(firms_frames),"power_files":len(power),"train":len(tr),"val":len(va),"test":len(te),"test_positives":int(te.y.sum()),"test_negatives":int((te.y==0).sum())},indent=2))
