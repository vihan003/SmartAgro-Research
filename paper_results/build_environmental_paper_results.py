from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROCESSED = ROOT / "data" / "environmental" / "processed"
ENV = ROOT / "environmental_monitoring"
OUT = ROOT / "paper_results" / "environmental_monitoring"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 11, "axes.labelsize": 9,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

DATASETS = {
    "predictive_irrigation": {
        "name": "Environmental Sensor Dataset for LSTM-Based Predictive Irrigation",
        "source": "https://data.mendeley.com/datasets/jtsyb593fy/1",
        "license": "CC BY 4.0",
        "target": "pump_status",
        "timestamp": "Original timestamp",
    },
    "motor_monitor": {
        "name": "Soil Moisture, Air Temperature, Humidity and Motor on/off Monitoring data",
        "source": "https://data.mendeley.com/datasets/fpdwmm7nrb/1",
        "license": "CC BY-NC 3.0",
        "target": "pump_status",
        "timestamp": "Synthetic 30-minute playback timestamps only",
    },
    "smartagro_iot": {
        "name": "Smart Agriculture and Plant Health Monitoring using IoT",
        "source": "https://data.mendeley.com/datasets/65jxyrxv7b/1",
        "license": "CC BY 4.0",
        "target": "None",
        "timestamp": "Timestamped multi-file archive readings",
    },
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (center-half, center+half)


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


frames = {}
summary_rows = []
sensor_rows = []

for key, info in DATASETS.items():
    df = pd.read_csv(PROCESSED / f"{key}.csv", parse_dates=["timestamp"])
    frames[key] = df
    numeric = df.select_dtypes(include=[np.number])
    missing_pct = float(df.isna().mean().mean()*100)
    dup = int(df.duplicated().sum())

    summary_rows.append({
        "dataset_key": key,
        "dataset_name": info["name"],
        "source_url": info["source"],
        "license": info["license"],
        "records": len(df),
        "processed_columns": len(df.columns),
        "numeric_features": len(numeric.columns),
        "sensor_variables": "; ".join([c for c in df.columns if c not in ("timestamp", "pump_status", "day", "time", "day_of_week", "hour", "moisture_change_1", "moisture_change_3", "temperature_rolling_3", "humidity_rolling_3", "soil_moisture_raw")]),
        "target": info["target"],
        "timestamp_availability": info["timestamp"],
        "timestamp_start": str(df.timestamp.min()) if "timestamp" in df else "Not available",
        "timestamp_end": str(df.timestamp.max()) if "timestamp" in df else "Not available",
        "missing_value_percent": missing_pct,
        "duplicate_rows": dup,
        "data_types": "; ".join(sorted(set(str(t) for t in df.dtypes))),
    })

    if info["target"] != "None":
        counts = df["pump_status"].value_counts().sort_index()
        summary_rows[-1]["class_0"] = int(counts.get(0, 0))
        summary_rows[-1]["class_1"] = int(counts.get(1, 0))
        summary_rows[-1]["class_1_percent"] = float(counts.get(1, 0)/len(df)*100)
    else:
        summary_rows[-1]["class_0"] = "N/A"
        summary_rows[-1]["class_1"] = "N/A"
        summary_rows[-1]["class_1_percent"] = "N/A"

    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s):
            q1, q3 = s.quantile([.25, .75])
            sensor_rows.append({
                "dataset_key": key, "variable": col, "unit_or_type": "percent" if "pct" in col or "humidity" in col or "moisture" in col else ("C" if "temperature" in col else "index" if "lux" in col or "light" in col else "binary" if col == "pump_status" else "derived"),
                "count": len(s), "missing_count": int(df[col].isna().sum()),
                "mean": s.mean(), "std": s.std(), "min": s.min(), "q1": q1,
                "median": s.median(), "q3": q3, "max": s.max(), "iqr": q3-q1,
            })

pd.DataFrame(summary_rows).to_csv(OUT / "dataset_summary.csv", index=False)
pd.DataFrame(sensor_rows).to_csv(OUT / "sensor_statistics.csv", index=False)

# Apply the existing rule system to datasets with compatible variables.
from environmental_monitoring.stress_detection import stress_for_row
stress_rows = []
for key, df in frames.items():
    for _, row in df.iterrows():
        r = stress_for_row(row.to_dict())
        stress_rows.append({"dataset_key": key, "timestamp": row.get("timestamp"),
                            "category": r["category"], "severity": r["severity"], "score": r["score"]})
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(OUT / "stress_category_records.csv", index=False)

# Core sensor distribution plots.
smart = frames["smartagro_iot"]
pred = frames["predictive_irrigation"]
motor = frames["motor_monitor"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
smart["temperature_c"].plot.hist(ax=axes[0,0], bins=20, color="#3977A8", edgecolor="white")
axes[0,0].set(title="SmartAgro IoT: air temperature", xlabel="Temperature (°C)", ylabel="Record count")
smart["humidity_pct"].plot.hist(ax=axes[0,1], bins=20, color="#27877B", edgecolor="white")
axes[0,1].set(title="SmartAgro IoT: relative humidity", xlabel="Humidity (%)", ylabel="Record count")
smart["soil_moisture_pct"].plot.hist(ax=axes[1,0], bins=20, color="#D89B2B", edgecolor="white")
axes[1,0].set(title="SmartAgro IoT: soil moisture", xlabel="Soil moisture (%)", ylabel="Record count")
smart["soil_ph"].plot.hist(ax=axes[1,1], bins=12, color="#D56D42", edgecolor="white")
axes[1,1].set(title="SmartAgro IoT: soil pH", xlabel="Soil pH", ylabel="Record count")
fig.tight_layout(); savefig(fig, "fig_environment_sensor_distributions")

fig, ax = plt.subplots(figsize=(10, 5.2))
sort = smart.sort_values("timestamp")
ax.plot(sort["timestamp"], sort["temperature_c"], label="Temperature", color="#D56D42", lw=1.8)
ax.plot(sort["timestamp"], sort["humidity_pct"], label="Humidity", color="#3977A8", lw=1.8)
ax.set(title="SmartAgro IoT: temperature and humidity recorded trends", xlabel="Recorded timestamp", ylabel="Sensor value (°C / %)")
ax.legend(frameon=False, ncol=2); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
savefig(fig, "fig_temperature_humidity_trends")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(smart.sort_values("timestamp")["timestamp"], smart.sort_values("timestamp")["soil_moisture_pct"], color="#27877B", lw=1.8)
axes[0].set(title="SmartAgro IoT soil moisture", xlabel="Recorded timestamp", ylabel="Soil moisture (%)"); axes[0].grid(alpha=.25)
axes[1].plot(pred.sort_values("timestamp")["timestamp"], pred.sort_values("timestamp")["soil_moisture_pct"], color="#3977A8", lw=.7)
axes[1].set(title="Predictive-irrigation soil moisture", xlabel="Recorded timestamp", ylabel="Soil moisture (%)"); axes[1].grid(alpha=.25)
fig.autofmt_xdate(); fig.tight_layout(); savefig(fig, "fig_soil_moisture_trends")

stress_count = stress_df[stress_df.dataset_key == "smartagro_iot"]["category"].value_counts()
order = ["Normal", "Low stress", "Moderate stress", "High stress"]
vals = [int(stress_count.get(k, 0)) for k in order]
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#27877B", "#D89B2B", "#D56D42", "#B65252"]
ax.bar(order, vals, color=colors)
for i,v in enumerate(vals): ax.text(i,v+1,str(v),ha="center",fontweight="bold")
ax.set(title="Existing rule-based stress categories: SmartAgro IoT archive", xlabel="Category", ylabel="Record count")
ax.grid(axis="y",alpha=.25); fig.tight_layout(); savefig(fig, "fig_stress_category_distribution")

# Dataset comparison and class distributions.
fig, ax = plt.subplots(figsize=(9, 5))
names = ["Predictive irrigation","Motor-monitor","SmartAgro IoT"]
rows = [len(pred),len(motor),len(smart)]
bars=ax.bar(names,rows,color=["#3977A8","#27877B","#D89B2B"])
for b,v in zip(bars,rows): ax.text(b.get_x()+b.get_width()/2,v+30,str(v),ha="center",fontweight="bold")
ax.set(title="Dataset record counts",ylabel="Processed records");ax.grid(axis="y",alpha=.25);fig.tight_layout();savefig(fig,"fig_dataset_record_counts")

fig, ax = plt.subplots(figsize=(8,5))
labels=["Pump off","Pump on"]
for i,(name,df) in enumerate([("Predictive irrigation",pred),("Motor-monitor",motor)]):
    c=df.pump_status.value_counts().reindex([0,1],fill_value=0)
    ax.bar([i-.18,i+.18],c.values,width=.35,label=name if i==0 else None)
    for j,v in enumerate(c.values): ax.text(j+i*.0+(-.18 if j==0 else .18),v+20,str(v),ha="center",fontsize=8)
ax.set_xticks([-.18,.18,0.82,1.18],["Predictive off","Predictive on","Motor off","Motor on"])
ax.set(title="Pump-status class distributions",ylabel="Record count");ax.grid(axis="y",alpha=.25);fig.tight_layout();savefig(fig,"fig_pump_class_distribution")

# Model verification from existing JSON.
eval_rows=[]
for key in ("predictive_irrigation","motor_monitor"):
    data=json.loads((ENV/"results"/f"{key}_metrics.json").read_text())
    tr=data["splits"]["train"]; va=data["splits"]["validation"]; te=data["splits"]["test"]
    df=frames[key]
    n=len(df)
    train_end=int(n*.70); val_end=train_end+int(n*.15)
    trdf=df.iloc[:train_end]; valdf=df.iloc[train_end:val_end]; tedf=df.iloc[val_end:] 
    for split_name, sdf in (("train",trdf),("validation",valdf),("test",tedf)):
        pass
    for model_name, metrics in data["models"].items():
        cm=metrics["confusion_matrix"]
        tn,fp,fn,tp=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        precision_pos=tp/(tp+fp) if tp+fp else np.nan
        recall_pos=tp/(tp+fn) if tp+fn else np.nan
        eval_rows.append({
            "dataset":key,"model":model_name,"train_rows":tr,"validation_rows":va,"test_rows":te,
            "train_class_0":int(trdf.pump_status.eq(0).sum()),"train_class_1":int(trdf.pump_status.eq(1).sum()),
            "validation_class_0":int(valdf.pump_status.eq(0).sum()),"validation_class_1":int(valdf.pump_status.eq(1).sum()),
            "test_class_0":int(tedf.pump_status.eq(0).sum()),"test_class_1":int(tedf.pump_status.eq(1).sum()),
            "accuracy":metrics["accuracy"],"precision_macro":metrics["precision_macro"],"recall_macro":metrics["recall_macro"],
            "f1_macro":metrics["f1_macro"],"f1_weighted":metrics["f1_weighted"],"roc_auc":metrics.get("roc_auc",np.nan),
            "pr_auc":metrics.get("pr_auc",np.nan),"positive_precision":precision_pos,"positive_recall":recall_pos,
            "tn":tn,"fp":fp,"fn":fn,"tp":tp,
            "evaluation_reliability":"Insufficient positive test cases (2 pump-on)" if key=="predictive_irrigation" else "Exploratory; source timestamps synthetic and label validity unverified",
        })
pd.DataFrame(eval_rows).to_csv(OUT/"evaluation_summary.csv",index=False)

# Existing model PR-AUC comparison plots.
for key in ("predictive_irrigation","motor_monitor"):
    d=pd.DataFrame([r for r in eval_rows if r["dataset"]==key])
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(d["model"].str.replace("_"," "),d["pr_auc"],color="#3977A8")
    ax.set_ylim(0,1.02);ax.set_ylabel("PR-AUC");ax.set_title(key.replace("_"," ")+": chronological/test PR-AUC")
    ax.tick_params(axis="x",rotation=25);ax.grid(axis="y",alpha=.25);fig.tight_layout();savefig(fig,"fig_"+key+"_pr_auc")

# Temporal stress/anomaly summary.
with pd.option_context("display.max_columns",None):
    pass

# Research text files.
dataset = pd.DataFrame(summary_rows)
stats = pd.DataFrame(sensor_rows)
evaldf = pd.DataFrame(eval_rows)
stress_counts = stress_df.groupby(["dataset_key","category"]).size().rename("count").reset_index()
stress_counts.to_csv(OUT/"stress_category_summary.csv",index=False)

smart_stats = stats[stats.dataset_key=="smartagro_iot"]
smart_temp = smart_stats[smart_stats.variable=="temperature_c"]
smart_humidity = smart_stats[smart_stats.variable=="humidity_pct"]
smart_moist = smart_stats[smart_stats.variable=="soil_moisture_pct"]
smart_ph = smart_stats[smart_stats.variable=="soil_ph"]


def vstat(var):
    r=smart_stats[smart_stats.variable==var].iloc[0]
    return r

t=r=vstat("temperature_c")
h=vstat("humidity_pct")
m=vstat("soil_moisture_pct")
p=vstat("soil_ph")
stress_values=stress_counts[stress_counts.dataset_key=="smartagro_iot"]
stress_text="; ".join(f"{r.category}: {int(r.count)}" for r in stress_values.itertuples())

research="""# Experimental Results: Environmental Monitoring and Crop-Stress Detection

## 3.1 Data description

Three public datasets were used. The predictive-irrigation dataset contained 960 timestamped observations collected at 30-minute intervals over 20 days. It included air temperature, relative humidity, soil moisture, light intensity, and a binary pump-status label. The motor-monitor dataset contained 3,000 source records with soil moisture, air temperature, air humidity, and pump-status information. Its original source CSV did not provide timestamps; timestamps were generated only for playback and split construction. The SmartAgro IoT sensor archive contained 101 matched timestamped observations for air temperature, relative humidity, light intensity, soil moisture, soil pH, and soil temperature, but no stress or pump-status ground-truth label.

No datasets were merged because their variable semantics and collection environments differed. The USDA/ARS Cook Agronomy Farm archive was downloaded and documented but was not included in the current monitoring or supervised analysis because it did not provide compatible air temperature, humidity, light, or validated stress labels.

## 3.2 Recorded sensor statistics

In the SmartAgro IoT archive, air temperature ranged from %.2f to %.2f °C (mean %.2f °C), relative humidity ranged from %.2f to %.2f%% (mean %.2f%%), soil moisture ranged from %.2f to %.2f%% (mean %.2f%%), and soil pH ranged from %.2f to %.2f (mean %.2f). The dashboard playback is an interpretation of these historical records, not live sensor acquisition.

## 3.3 Rule-based environmental stress analysis

The existing configurable rule system was applied to the SmartAgro IoT records. The resulting categories were: %s. These classes are rule-generated labels and must not be interpreted as diagnosed crop stress or validated agronomic outcomes.

## 3.4 Machine-learning evaluation

For pump-status prediction, the predictive-irrigation dataset contained 932 pump-off and 28 pump-on observations. A chronological 70%%/15%%/15%% split produced 672 training, 144 validation, and 144 test observations. The test set contained 142 pump-off and only 2 pump-on observations. All existing models returned perfect point estimates on this test slice, but the positive class support is insufficient for a statistically meaningful claim. Precision, recall, ROC-AUC, PR-AUC, and confidence in generalization are therefore not reported as reliable field performance.

The motor-monitor dataset contained 1,431 pump-off and 1,569 pump-on observations. Chronological-style partitioning produced 2,100 training, 450 validation, and 450 test observations. The test set contained 208 pump-off and 242 pump-on observations, and the existing models showed perfect or near-perfect test results. This is an exploratory in-dataset result only: the source lacks original timestamps, the timestamps are synthetic for playback, and label validity/field conditions are not established. It should not be presented as a validated irrigation or crop-stress model.

## 3.5 Scientific interpretation and limitations

The implementation demonstrates preprocessing, historical playback, interpretable rule triggering, and pump-status model execution on public datasets. It does not demonstrate live IoT deployment, physical sensor communication, crop-specific stress diagnosis, or field-level agricultural effectiveness. Rule thresholds are configurable and crop/soil-dependent; they require agronomic and field validation. The SmartAgro archive has no valid crop-stress target, while the pump-status datasets provide irrigation motor labels rather than crop-stress ground truth.

## Research-ready subsection

Environmental monitoring was evaluated using publicly available, timestamped or timestamp-reconstructable sensor records. The SmartAgro IoT archive was used for recorded playback and rule-based stress analysis. In 101 matched readings, air temperature ranged from %.2f–%.2f °C (mean %.2f °C), relative humidity ranged from %.2f–%.2f%% (mean %.2f%%), soil moisture ranged from %.2f–%.2f%% (mean %.2f%%), and soil pH ranged from %.2f–%.2f (mean %.2f). The existing rule engine produced %s. These categories were rule-generated and were not validated as crop-stress diagnoses.

Pump-status classification was assessed separately on two datasets. On the predictive-irrigation dataset, the chronological test set contained only 2 pump-on observations among 144 test readings, so reported perfect scores were excluded from research claims. On the motor-monitor dataset, the test set contained 208 pump-off and 242 pump-on observations; perfect in-dataset metrics were retained only as exploratory, because original timestamps and field ground-truth validity were unavailable. Accordingly, this work establishes a reproducible software pipeline and recorded-data analysis, but not a clinically or agronomically validated stress-monitoring system.
"""

research=research % (t["min"],t["max"],t["mean"],h["min"],h["max"],h["mean"],m["min"],m["max"],m["mean"],p["min"],p["max"],p["mean"],stress_text,
                    t["min"],t["max"],t["mean"],h["min"],h["max"],h["mean"],m["min"],m["max"],m["mean"],p["min"],p["max"],p["mean"],stress_text)

(OUT/"environmental_results.md").write_text(research,encoding="utf-8")
(OUT/"research_paper_section.md").write_text(research.split("## Research-ready subsection\n\n",1)[-1].strip(),encoding="utf-8")

repro="""# Reproducibility

## Environment

Use the project virtual environment:

```powershell
cd D:\\agro2
.\\.venv\\Scripts\\python.exe
```

## Reproduce preprocessing

```powershell
.\\.venv\\Scripts\\python.exe environmental_monitoring\\preprocess.py
```

## Reproduce original model evaluation

```powershell
.\\.venv\\Scripts\\python.exe environmental_monitoring\\train_models.py
```

## Reproduce this paper-analysis package

```powershell
.\\.venv\\Scripts\\python.exe paper_results\\build_environmental_paper_results.py
```

## Inputs

- `data/environmental/processed/predictive_irrigation.csv`
- `data/environmental/processed/motor_monitor.csv`
- `data/environmental/processed/smartagro_iot.csv`
- `environmental_monitoring/results/*_metrics.json`
- `environmental_monitoring/datasets/dataset_metadata.json`

## Procedures

- Processed numerical missing values are excluded or interpolated as documented in `preprocess.py`.
- Stress categories use `environmental_monitoring/stress_detection.py` with its default configurable thresholds.
- The original model evaluation uses a 70%/15%/15% chronological-style split for both datasets. The motor-monitor timestamps are synthetic and not real chronology.
- All figures are regenerated at 300 DPI PNG and PDF.
"""
(OUT/"reproducibility.md").write_text(repro,encoding="utf-8")

print(json.dumps({"out":str(OUT),"datasets":len(frames),"stress_records":len(stress_df),"sensor_statistics":len(sensor_rows),"evaluation_rows":len(eval_rows)},indent=2))
