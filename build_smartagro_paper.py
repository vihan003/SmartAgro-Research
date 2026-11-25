from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

MOBILE_CSV = OUT / "results_a" / "per_class_grouped__mobilenetv2.csv"
CNN_CSV = OUT / "results_a" / "per_class_grouped__small_cnn.csv"
CM_MOBILE = OUT / "results_a" / "cm_grouped__mobilenetv2.png"
CM_CNN = OUT / "results_a" / "cm_grouped__small_cnn.png"
FIRE_JSON = OUT / "results_c" / "results_c.json"

mobile = pd.read_csv(MOBILE_CSV, index_col=0)
small = pd.read_csv(CNN_CSV, index_col=0)
fire = json.loads(FIRE_JSON.read_text(encoding="utf-8"))
models = fire["models"]
setup = fire["setup"]

COLORS = {
    "navy": "#18324B",
    "blue": "#3977A8",
    "teal": "#27877B",
    "green": "#4A8C67",
    "amber": "#D89B2B",
    "orange": "#D56D42",
    "red": "#B65252",
    "gray": "#687887",
    "light": "#EEF3F5",
    "ink": "#202C36",
    "muted": "#56636D",
    "line": "#CBD5DC",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["line"],
    "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"],
})


def save_figure(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, w, h, title, body, color, edge=None, title_color="white", body_color="white", fs=9):
    x, y = xy
    shape = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor=edge or color,
        linewidth=1.2,
    )
    ax.add_patch(shape)
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", color=title_color,
            fontsize=fs, weight="bold", wrap=True)
    ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center", color=body_color,
            fontsize=max(fs - 1.1, 6.4), wrap=True, linespacing=1.25)
    return shape


def arrow(ax, start, end, color=None, lw=1.6, style="-|>"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": style, "color": color or COLORS["gray"], "lw": lw,
                            "shrinkA": 1, "shrinkB": 1})


def make_architecture():
    fig, ax = plt.subplots(figsize=(11.7, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .965, "SmartAgro framework and evidence boundary", ha="center", va="top",
            fontsize=15, weight="bold", color=COLORS["navy"])
    ax.text(.5, .915, "Three independent input streams; the shared decision layer is proposed and unvalidated",
            ha="center", va="top", fontsize=9.5, color=COLORS["muted"])
    lanes = [
        (.72, "Leaf images", "RGB leaf image\n16-class result artifacts\nProvisional", COLORS["amber"]),
        (.48, "Field sensors", "Temperature · humidity\nSoil moisture\nNo paired field records", COLORS["gray"]),
        (.24, "Satellite + weather", "VIIRS hotspot + POWER\nRetrospective next-day\nranking replayed", COLORS["teal"]),
    ]
    for y, title, body, color in lanes:
        box(ax, (.025, y), .21, .14, title, body, color, fs=9.4)
    box(ax, (.305, .72), .23, .14, "Image classifier", "MobileNetV2 and small CNN\nReported subset comparison", COLORS["amber"], fs=9)
    box(ax, (.305, .48), .23, .14, "Stress estimator", "Crop-stage and sensor fusion\nProposed; no model evaluated", COLORS["gray"], fs=9)
    box(ax, (.305, .24), .23, .14, "Regional detector model", "Logistic regression / boosted trees\nCache-reproduced on grid cell-days", COLORS["teal"], fs=9)
    box(ax, (.61, .46), .18, .23, "Evidence-aware\npolicy layer", "Time, location, source,\nconfidence and uncertainty\n\nProposed", COLORS["navy"], fs=9)
    box(ax, (.84, .46), .14, .23, "User interface", "Risk-tiered notices\nDashboard and\naction follow-up\n\nProposed", COLORS["blue"], fs=8.7)
    for y in (.79, .55, .31):
        arrow(ax, (.235, y), (.305, y), COLORS["gray"])
        arrow(ax, (.535, y), (.61, .575), COLORS["gray"])
    arrow(ax, (.79, .575), (.84, .575), COLORS["blue"])
    ax.text(.5, .08, "Evidence key", weight="bold", color=COLORS["navy"], ha="center", fontsize=9)
    for x, c, label in [(.17, COLORS["teal"], "Cache metrics replayed"),
                         (.43, COLORS["amber"], "Disease metrics provisional"),
                         (.72, COLORS["gray"], "Proposed / not evaluated")]:
        ax.add_patch(patches.Rectangle((x, .035), .022, .022, facecolor=c, transform=ax.transAxes, clip_on=False))
        ax.text(x + .03, .046, label, transform=ax.transAxes, va="center", fontsize=7.6, color=COLORS["muted"])
    save_figure(fig, "fig01_smartagro_architecture")


def make_workflow():
    fig, ax = plt.subplots(figsize=(11.7, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .96, "Evidence workflow and current execution state", ha="center", va="top",
            fontsize=14, weight="bold", color=COLORS["navy"])
    xs = [.035, .235, .435, .635, .835]
    widths = [.15, .15, .15, .15, .13]
    labels = [
        ("Inventory", "Local data, scripts,\nreports and references", COLORS["blue"], "Audited"),
        ("Disease artifacts", "CSV summaries + 16×16\nPNG matrices", COLORS["amber"], "Metrics provisional"),
        ("Fire cache", "91 FIRMS files + 54\nPOWER point files", COLORS["teal"], "Inputs replayed"),
        ("Metric replay", "All reported AP and\nthreshold metrics match", COLORS["teal"], "Exact reproduction"),
        ("Field validation", "Paired image, sensors,\nlabels and actions", COLORS["gray"], "Future work"),
    ]
    y = .48; h = .25
    for i, (title, body, color, status) in enumerate(labels):
        box(ax, (xs[i], y), widths[i], h, title, body, color, fs=8.4)
        ax.text(xs[i] + widths[i] / 2, .37, status, ha="center", va="center", fontsize=8.2,
                color=color, weight="bold")
        if i < len(labels) - 1:
            arrow(ax, (xs[i] + widths[i], y + h / 2), (xs[i + 1] - .012, y + h / 2), COLORS["gray"])
    ax.text(.5, .18,
            "The historical status report predates the output artifacts. Its ‘no valid metrics’ statement is stale for Experiment C;\n"
            "the disease result files remain unlinked to a run manifest or prediction-level output.",
            ha="center", va="center", fontsize=9, color=COLORS["muted"], linespacing=1.5)
    save_figure(fig, "fig02_evidence_workflow")


def make_cm(source: Path, stem: str, title: str):
    img = Image.open(source).convert("RGB")
    fig = plt.figure(figsize=(8.5, 7.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.7], hspace=.12)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, weight="bold", color=COLORS["navy"], pad=4)
    ax.text(-.045, .5, "True class index", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=8, color=COLORS["muted"])
    ax.text(.5, -.04, "Predicted class index", transform=ax.transAxes,
            va="center", ha="center", fontsize=8, color=COLORS["muted"])
    legend_ax = fig.add_subplot(gs[1]); legend_ax.axis("off")
    names = [str(x) for x in mobile.index if x not in ("accuracy", "macro avg", "weighted avg")]
    for idx, name in enumerate(names):
        col = idx // 8
        row = idx % 8
        x = .015 + col * .50
        y = .92 - row * .125
        nice = name.replace("___", " — ").replace("_", " ")
        legend_ax.text(x, y, f"{idx:>2}  {nice}", transform=legend_ax.transAxes,
                       ha="left", va="center", fontsize=6.8, color=COLORS["ink"])
    legend_ax.text(.99, .01,
                   "Class order follows the checked-in sorted-label script and is listed above. Source PNG contains no numeric cell labels or color scale.",
                   transform=legend_ax.transAxes, ha="right", va="bottom", fontsize=6.1, color=COLORS["muted"])
    save_figure(fig, stem)


def make_disease_comparison():
    acc = [100 * mobile.loc["accuracy", "precision"], 100 * small.loc["accuracy", "precision"]]
    f1 = [100 * mobile.loc["macro avg", "f1-score"], 100 * small.loc["macro avg", "f1-score"]]
    x = np.arange(2); width = .34
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color=COLORS["blue"])
    ax.bar(x + width / 2, f1, width, label="Macro-F1", color=COLORS["orange"])
    for xs, vals in [(x - width / 2, acc), (x + width / 2, f1)]:
        for xx, yy in zip(xs, vals):
            ax.text(xx, yy + 1.2, f"{yy:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["MobileNetV2", "Small CNN"])
    ax.set_ylim(0, 108); ax.set_ylabel("Score (%)")
    ax.set_title("Reported disease-classification scores (16-class subset)", color=COLORS["navy"], weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color=COLORS["line"], alpha=.55); ax.set_axisbelow(True)
    ax.text(.5, -.22, "Provisional: score files lack a run manifest and prediction-level outputs.",
            transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["muted"])
    save_figure(fig, "fig05_disease_model_comparison")


def make_fire_ap():
    order = ["persistence (fire yesterday)", "climatology (day-of-season rate)",
             "weather index (hot, dry, windy)", "logistic regression", "gradient-boosted trees"]
    labels = ["Persistence", "Climatology", "Weather index", "Logistic regression", "Gradient-boosted trees"]
    vals = [models[k]["AUPRC"] for k in order]
    lows = [models[k]["AUPRC_CI95_block_bootstrap"][0] for k in order]
    highs = [models[k]["AUPRC_CI95_block_bootstrap"][1] for k in order]
    err = np.array([[v - lo for v, lo in zip(vals, lows)], [hi - v for v, hi in zip(vals, highs)]])
    cols = [COLORS["gray"], COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["teal"]]
    fig, ax = plt.subplots(figsize=(9, 5.1))
    x = np.arange(len(order))
    ax.bar(x, vals, color=cols, width=.68, yerr=err, capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": COLORS["ink"]})
    ax.axhline(setup["base_rate_test"], color=COLORS["red"], linestyle="--", linewidth=1.5,
               label=f"Test positive prevalence ({setup['base_rate_test']:.3f})")
    for xx, v in zip(x, vals): ax.text(xx, v + .025, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("Average precision (AUPRC)")
    ax.set_title("Regional next-day VIIRS detection ranking", color=COLORS["navy"], weight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=COLORS["line"], alpha=.55); ax.set_axisbelow(True)
    ax.text(.5, -.27, "Error bars: supplied 95% weekly block-bootstrap intervals (200 resamples).",
            transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["muted"])
    save_figure(fig, "fig06_fire_auprc")


def make_fire_tradeoffs():
    order = ["persistence (fire yesterday)", "climatology (day-of-season rate)",
             "weather index (hot, dry, windy)", "logistic regression", "gradient-boosted trees"]
    short = ["Persistence", "Climatology", "Weather index", "Logistic regression", "Boosted trees"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["teal"], COLORS["green"]]
    alert = np.array([models[k]["alert_share_pct"] for k in order])
    recall = np.array([100 * models[k]["recall"] for k in order])
    precision = np.array([100 * models[k]["precision"] for k in order])
    false = np.array([models[k]["false_alarms_per_100_negative_cell_days"] for k in order])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [1.18, 1]})
    offsets = [(0, 10), (-8, -14), (0, 10), (-38, -14), (8, 9)]
    for i, (x, y, c) in enumerate(zip(alert, recall, colors)):
        left.scatter(x, y, s=68, color=c, edgecolor="white", linewidth=.8, zorder=3)
        left.annotate(short[i], (x, y), xytext=offsets[i], textcoords="offset points", fontsize=7.2, color=COLORS["ink"])
    left.set_xlim(30, 88); left.set_ylim(48, 91)
    left.set_xlabel("Test cell-days alerted (%)")
    left.set_ylabel("Test detection recall (%)")
    left.set_title("Coverage versus detections captured", color=COLORS["navy"], weight="bold", fontsize=10)
    left.grid(color=COLORS["line"], alpha=.55); left.set_axisbelow(True)
    y = np.arange(len(order))
    right.barh(y, false, color=colors, height=.62)
    right.set_yticks(y, short); right.invert_yaxis()
    right.set_xlabel("False alarms per 100 negative cell-days")
    right.set_title("False-alarm burden at selected threshold", color=COLORS["navy"], weight="bold", fontsize=10)
    right.grid(axis="x", color=COLORS["line"], alpha=.55); right.set_axisbelow(True)
    for yy, value, p in zip(y, false, precision):
        right.text(value + 1.2, yy, f"{value:.1f}  |  precision {p:.1f}%", va="center", fontsize=7)
    right.set_xlim(0, max(false) + 31)
    fig.suptitle("Operational test-set trade-offs after validation threshold selection", y=1.02,
                 fontsize=12, color=COLORS["navy"], weight="bold")
    fig.text(.5, -.03, "Each model's threshold was selected on validation data to target about 80% recall; plotted recall is from test data.",
             ha="center", fontsize=7.8, color=COLORS["muted"])
    fig.tight_layout()
    save_figure(fig, "fig07_fire_operational_tradeoffs")


def make_fusion():
    fig, ax = plt.subplots(figsize=(11.7, 6.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .96, "Proposed multimodal field-validation architecture", ha="center", va="top",
            fontsize=14, weight="bold", color=COLORS["navy"])
    sources = [
        (.035, .72, "Plant image", "Timestamp + plot ID\nCrop / symptom label", COLORS["blue"]),
        (.035, .49, "Sensor stream", "Calibrated temperature, RH\nSoil water and wetness", COLORS["green"]),
        (.035, .26, "Regional context", "Forecast inputs + VIIRS\nLabel source and latency", COLORS["teal"]),
    ]
    for x, y, title, body, color in sources: box(ax, (x, y), .22, .14, title, body, color, fs=8.5)
    box(ax, (.305, .43), .22, .24, "Time / space alignment", "UTC time\nFarm / plot IDs\nQuality / missingness\n\nAbsent in supplied data", COLORS["amber"], fs=8.3)
    box(ax, (.60, .62), .18, .19, "Component models", "Image-only\nSensor-only\nFused model\nTo be evaluated", COLORS["navy"], fs=8.2)
    box(ax, (.60, .30), .18, .19, "Risk policy", "Crop-stage rules\nUncertainty / abstain\nAgronomist tiers\nProposed", COLORS["gray"], fs=8.0)
    box(ax, (.83, .43), .15, .24, "Action record", "Alert delivered\nFarmer response\nAction confirmed\nOutcome label\nFuture", COLORS["orange"], fs=7.8)
    for y in (.79, .56, .33): arrow(ax, (.255, y), (.305, .55), COLORS["gray"])
    arrow(ax, (.525, .55), (.60, .715), COLORS["gray"])
    arrow(ax, (.69, .62), (.69, .49), COLORS["gray"])
    arrow(ax, (.78, .395), (.83, .54), COLORS["gray"])
    ax.text(.5, .10, "No synchronized image–sensor observations, confirmed stress labels, action logs, or field outcomes were supplied.",
            ha="center", va="center", fontsize=8.6, color=COLORS["red"], weight="bold")
    save_figure(fig, "fig08_proposed_multimodal_fusion")


def make_roadmap():
    fig, ax = plt.subplots(figsize=(11.7, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .96, "Evidence-to-validation roadmap", ha="center", va="top",
            fontsize=14, weight="bold", color=COLORS["navy"])
    stages = [
        ("A. Provenance audit", "Completed\nInventory, contradictions,\nsource-file checks", COLORS["teal"]),
        ("B. Disease benchmark", "Reported, unverified\n16-class summaries; no run\nmanifest or predictions", COLORS["amber"]),
        ("C. Fire replay", "Reproduced from cache\nExact metric match; source\nchecksums not recorded", COLORS["teal"]),
        ("D. Paired field fusion", "Not evaluated\nNo synchronized sensors,\nimages or stress labels", COLORS["gray"]),
        ("E. Prospective use", "Future validation\nExternal farms, forecast inputs,\naction and outcome study", COLORS["blue"]),
    ]
    x0=.03; gap=.018; w=.176; y=.46; h=.27
    for i,(title,body,color) in enumerate(stages):
        x=x0+i*(w+gap)
        box(ax,(x,y),w,h,title,body,color,fs=8.1)
        if i < len(stages)-1: arrow(ax,(x+w,y+h/2),(x+w+gap-.003,y+h/2),COLORS["gray"])
    ax.text(.5,.25,"Current stopping point: separate component evidence supports a transparent prototype paper, not a field-tested integrated product.",
            ha="center",va="center",fontsize=9,color=COLORS["muted"],wrap=True)
    ax.text(.5,.12,"Evidence states distinguish an exact offline replay from a metric file whose experiment provenance remains unresolved.",
            ha="center",va="center",fontsize=8.2,color=COLORS["muted"])
    save_figure(fig,"fig09_evidence_roadmap")


make_architecture()
make_workflow()
make_cm(CM_MOBILE, "fig03_mobilenetv2_confusion_matrix", "MobileNetV2: supplied 16 × 16 confusion-matrix image")
make_cm(CM_CNN, "fig04_smallcnn_confusion_matrix", "Small CNN: supplied 16 × 16 confusion-matrix image")
make_disease_comparison()
make_fire_ap()
make_fire_tradeoffs()
make_fusion()
make_roadmap()


def pct(v, digits=2):
    return f"{100 * float(v):.{digits}f}%"


def hcell(s, cls=""):
    return f'<td class="{cls}">{s}</td>'


def table(caption, headers, rows, cls=""):
    head = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(hcell(str(x)) for x in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><p class="tablecap"><b>{caption}</b></p><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(stem, caption, cls=""):
    return f'<figure class="{cls}"><img src="../figures/{stem}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def h2(n, title):
    return f'<h2>{n}. {title}</h2>'


parts = []
parts.append('''
<section class="titlepage">
<p class="eyebrow">Research article · evidence-audited component study</p>
<h1>SmartAgro for Agricultural Monitoring: Evidence-Audited Component Results and a Validation Framework for Crop Disease, Environmental Stress, and Regional Fire-Risk Assessment</h1>
<p class="authors">[Author name(s) to be confirmed]</p>
<p class="affiliation">[Affiliation(s) to be confirmed] · [Corresponding author email]</p>
<p class="status"><b>Evidence status.</b> Disease scores are reported artifact values whose run provenance is unresolved. Fire-risk scores are reproduced exactly from the supplied cached inputs and checked-in code, but the input files lack request logs and checksums. Environmental-stress sensing, multimodal fusion, and preventive action verification have not been empirically evaluated.</p>
<h2 class="abstracthead">Abstract</h2>
<p class="abstract">Agricultural monitoring systems often treat leaf disease, environmental stress, and fire as separate prediction tasks, although farm decisions depend on their timing and context. This paper presents SmartAgro as a modular decision-support framework and audits the evidence currently available for its components. The audit found a local PlantVillage color-image tree with 38 class folders and 54,305 images, while the disease result files cover 16 classes and 3,563 test examples. Their CSV summaries report MobileNetV2 accuracy of 97.61% and macro-F1 of 96.92%, compared with 75.75% and 65.35% for a small CNN. The class supports and summary arithmetic are internally coherent, but no run manifest, prediction-level output, or numeric confusion-matrix data link these scores to the checked-in training script; they are therefore provisional. In contrast, the fire-risk cache contains 91 VIIRS/S-NPP CSV pulls and 54 NASA POWER grid-point files. Replaying these files through the supplied pipeline reproduces all reported test metrics exactly. On a 2022–2023 temporal holdout, gradient-boosted trees achieve average precision 0.8723 and logistic regression 0.8530, compared with 0.7409 for persistence, 0.5452 for day-of-season climatology, and 0.4369 for the script-defined weather index. The target is next-day non-low-confidence satellite hotspot detection in a regional grid, not confirmed crop-residue burning. The test positive rate is 49.54%, and model differences have overlapping weekly block-bootstrap intervals. No paired field sensor records, confirmed stress labels, multimodal evaluation, action logs, or field outcomes were supplied. The contribution is a reproducible component-level analysis and a concrete validation design; the integrated SmartAgro system remains proposed.</p>
<p class="keywords"><b>Keywords:</b> plant disease classification; PlantVillage; environmental stress; agricultural IoT; VIIRS; NASA POWER; hotspot detection; precision–recall; evidence provenance; decision support</p>
<p class="counting">Counting convention: the target main-manuscript count runs from title/front matter through the conclusion; references and appendices are excluded. The final rendered count is reported in the accompanying quality review.</p>
</section>
''')

parts.append(h2(1, "Introduction"))
parts.append('''
<p>Smallholder and commercial farms face overlapping biological and environmental hazards. A lesion on a leaf may indicate infection, nutrient stress, or damage; dry soil and high atmospheric demand can compound heat stress; and a detected thermal anomaly may require a response that differs from irrigation or disease treatment. A useful digital advisory system must therefore do more than place several classifiers behind one interface. It must define what each input can establish, when an alert is actionable, and how uncertainty is conveyed to the people expected to use it.</p>
<p>Image-based plant disease classification is a mature research area, but headline performance is sensitive to how images are collected and divided. Mohanty et al. trained convolutional networks on 54,306 controlled-condition PlantVillage images and reported 99.35% accuracy, while their evaluation on small sets of externally sourced images showed a substantial drop in accuracy [1]. The contrast illustrates a central issue for agricultural computer vision: a held-out image from the same collection does not automatically represent a new farm, cultivar, camera, season, or symptom stage. Dataset provenance and split design are therefore part of the scientific claim, not merely implementation details.</p>
<p>Environmental sensors can add information that is absent from a leaf image, but their value depends on synchronizing measurements to the plant and event being classified. Air temperature, relative humidity, soil-water status, leaf wetness, and crop stage can describe exposure or context; they do not by themselves confirm a disease or prove that a crop is stressed. The supplied materials contain no paired image–sensor series or confirmed field stress labels. Consequently, SmartAgro's environmental and fusion components can be specified and designed for validation, but their predictive performance cannot be estimated from the available evidence.</p>
<p>Fire assessment creates a different measurement problem. NASA FIRMS distributes active-fire and thermal-anomaly detections derived from satellite observations. A detection is a useful regional observation, but it is not an independently verified fire cause, a cropland label, or a field-scale alarm. The VIIRS product has a nominal 375 m pixel footprint, yet this project aggregates detections and meteorology into much larger daily grid-cell units. It predicts whether a cell has a non-low-confidence detection on the following day. That outcome can reflect persistence of an ongoing thermal event and must not be described as confirmed ignition or crop-residue burning [6], [7].</p>
<p>The present study narrows the broad SmartAgro proposal into three evidence questions. First, what can be established from the available disease-classification artifacts, and what remains unverified? Second, can the reported fire-risk metrics be reproduced from the supplied cache and source code, and what do those metrics measure? Third, what evidence would be required to evaluate environmental stress sensing, image–sensor fusion, and preventive actions as one agricultural decision process? These questions make the paper an evidence-audited component study rather than a report of an integrated field deployment.</p>
<p>The practical motivation remains important. A grower, agronomist, or extension service may benefit from a dashboard that combines diagnostic information with crop stage, weather, and regional hazard signals. However, the decision layer must preserve source identity and spatial scale. A disease classifier's output is a model prediction from an image; a sensor stream is a measurement with calibration error; a satellite hotspot is an observed thermal anomaly; and a recommendation is a policy choice. Keeping those objects distinct is necessary for trustworthy use and for a future evaluation of whether the system improves decisions.</p>
''')

parts.append(h2(2, "Background and related work"))
parts.append('''
<p><b>Plant disease classification and domain shift.</b> PlantVillage enabled large-scale image classification research across multiple crops and diseases [1]. Its controlled acquisition conditions support repeatable benchmark work, but do not reproduce the clutter, occlusion, changing illumination, mixed symptoms, and device variation of routine field photography. PlantDoc was assembled from internet images specifically to provide a more varied visual setting; its authors describe 2,598 images across 13 plant species and up to 17 disease classes [2]. The datasets answer different questions. A score on PlantVillage estimates performance under that collection's image distribution, while field-like data are needed to estimate transfer to natural backgrounds and real diagnostic workflows. Neither dataset supplies synchronized sensor observations for this study.</p>
<p><b>Convolutional networks and transfer learning.</b> MobileNetV2 uses inverted residual blocks and linear bottlenecks to reduce computation for mobile vision models [3]. A compact CNN trained from scratch is a useful reference for estimating the benefit of an ImageNet-initialized feature extractor in a fixed task, but the comparison is not a universal test of architecture families. The checked-in experiment compares one small CNN with one frozen MobileNetV2 backbone under a single code path. It does not establish that transfer learning is always superior, nor does it isolate pretraining from model capacity, optimization, and representation differences. A fair general claim would require repeated, controlled runs and comparable compute and tuning budgets.</p>
<p><b>Structured validation.</b> Agricultural images and cell-day records have dependence structures. Near-identical views of one leaf can cross a random split; nearby places and dates can share weather, management, and fire activity. Random partitioning may therefore estimate interpolation among familiar observations rather than generalization to unseen leaves, sites, or years. The checked-in disease script proposes approximate perceptual-hash grouping followed by stratified group folds. The fire script instead makes a forward temporal split while retaining the same spatial cells. These are different validation targets. General guidance for structured ecological data recommends choosing blocks to match the intended prediction domain and warns that blocking itself changes the extrapolation problem [4]. Neither split alone supports deployment across new farms and new years.</p>
<p><b>Satellite fire products and weather fields.</b> The VIIRS 375 m active-fire product was designed to detect active thermal anomalies using a contextual algorithm [6]. FIRMS cautions that detected activity may arise from fires, hot smoke, agriculture, or other sources; clouds and satellite overpass timing also cause omissions [7]. The distinction matters because a model trained on FIRMS-derived labels inherits both the product's measurement process and its omissions. NASA POWER supplies daily meteorological variables based on gridded source data; its meteorological fields originate from MERRA-2/GEOS products at approximately 0.5° latitude by 0.625° longitude [8], [9]. POWER daily requests default to local solar time unless UTC is requested explicitly [8]. Retrospective observed or reanalysis weather is not an operational weather forecast, and its date must be aligned to satellite observations before modeling.</p>
<p><b>Evaluation under different positive rates.</b> Average precision summarizes ranking across score thresholds. Its interpretation depends on the positive-class frequency; a random ranking has expected precision–recall performance tied to prevalence, and a single ranking score does not specify an alert threshold or workload [5]. Operational systems also require precision, recall, missed-event rate, false alarms, and the fraction of locations alerted at a chosen decision threshold. The fire results provide such quantities for thresholds selected on a validation year, making it possible to discuss a concrete trade-off. They do not include response-time, forecast availability, or action outcomes.</p>
<p>Across these literatures, a recurring methodological requirement is to match the claim to the unit and context of evaluation. A leaf-image benchmark is not a crop diagnosis trial; a satellite grid-cell detection is not a farm-level fire forecast; and an architecture diagram is not evidence that sensor fusion works. This paper uses those distinctions to structure both the results and the future validation plan.</p>
''')

parts.append(h2(3, "Research gap, questions, and contributions"))
parts.append('''
<p>The available draft proposed image classification, sensor monitoring, satellite fire assessment, and a shared recommendation engine. The present audit shows that only the image and satellite components have numerical artifacts, and those artifacts have unequal evidential status. Disease metrics are internally consistent but lack run provenance. Fire metrics are exactly reproducible from local cache files but lack source-response checksums and have a time-standard alignment risk. No paired sensor or action records support evaluation of the integrated layer. The defensible gap is therefore not that no one has ever combined these technologies; this paper does not claim that. The gap addressed here is the absence of an explicit, auditable bridge between component results and an integrated agricultural decision-support claim in this project.</p>
<p><b>RQ1.</b> What can the supplied disease artifacts establish about a reported 16-class MobileNetV2 versus small-CNN comparison, and which claims remain unverified?</p>
<p><b>RQ2.</b> Can the regional next-day VIIRS detection metrics be reproduced from the cached FIRMS and POWER inputs, and how do model ranking and alert burden compare on the supplied temporal test period?</p>
<p><b>RQ3.</b> What data, labels, baselines, and validation design are required before environmental sensing, multimodal fusion, and preventive actions can be evaluated as SmartAgro capabilities?</p>
<p>The contributions are fourfold. First, we inventory the manuscript, scripts, result files, image data, and satellite/weather cache and reconcile a stale experiment-status report against later outputs. Second, we recompute fire-model results from the saved source files using the checked-in processing and evaluation code and obtain exact metric agreement. Third, we report disease metrics only as provisional artifact values, retaining their full class support and making their missing provenance explicit. Fourth, we define an evidence-tagged architecture and a staged validation protocol that separates image diagnosis, sensor-based stress estimation, regional hotspot detection, and decision-policy evaluation.</p>
<p>These contributions are methodological and evidential. They do not establish a new model architecture, a new public dataset, a statistically significant superiority result, or an integrated product that has been tested with farmers. Table 1 maps each research objective to the evidence available and the claim that is currently defensible.</p>
''')

rows = [
    ["RQ1 · disease", "16-class per-class CSVs, aggregate rows, and two 16 × 16 PNGs", "Reported accuracy and macro-F1 are numerically consistent with the CSVs; support totals 3,563", "Provisional: no run manifest, predictions, or numeric confusion matrices"],
    ["RQ2 · satellite detection", "Main results JSON, run log, 91 FIRMS CSVs, 54 POWER JSON files, checked-in script", "Every reported AP and threshold metric is reproduced exactly from cache", "Regional next-day non-low-confidence VIIRS detection only; date-basis and source provenance remain limited"],
    ["RQ3 · sensors and fusion", "Framework proposal and sensor-variable discussion", "A validation design can be specified", "No paired sensor/image data, confirmed stress labels, action records, or fusion model"],
]
parts.append(table("Table 1. Research questions and evidence status.", ["Objective", "Available evidence", "Supported conclusion", "Boundary"], rows, "compact"))

parts.append(h2(4, "SmartAgro system architecture and evidence boundaries"))
parts.append('''
<p>SmartAgro is specified as a modular pipeline with three data streams and a proposed policy layer. The first stream accepts a leaf photograph and returns a crop–disease class distribution. The supplied disease script implements two candidate classifiers: a small CNN and a MobileNetV2 feature extractor. The available result files cover only a 16-class subset and are not linked to a retained prediction file. That stream is therefore marked reported but unverified.</p>
<p>The second stream is intended to collect local environmental measurements. A sensor gateway would attach timestamps, plot identifiers, device identifiers, calibration metadata, and missing-data flags to each observation. The current workspace does not contain such readings. No stress probability, irrigation recommendation, or sensor-based disease adjustment is supported as a measured capability. In the architecture this branch is a design requirement, not a tested service.</p>
<p>The third stream joins satellite hotspots with gridded weather. In the supplied component experiment, the unit of analysis is a grid cell on a calendar date. The code joins VIIRS detections and NASA POWER fields, constructs lagged detections and weather summaries, and scores next-day detection. The saved inputs and model pipeline reproduce the result JSON, so this component has stronger computational reproducibility than the disease result. Its target remains a satellite-derived label and its test is temporal at known grid cells.</p>
<p>A policy layer is proposed to combine these outputs only after each source is represented with its own semantics. The record should include event time, ingestion time, geographic support, units, model version, score calibration, and evidence status. An image class distribution must not be converted directly into a treatment instruction without crop and symptom review. A regional hotspot score should not be presented as a field alert. A water-stress signal should not be treated as fire suppression. The layer should abstain or request human review when required inputs are absent or model confidence is not calibrated.</p>
<p>Figure 1 distinguishes the component with a reproducible offline replay, the disease metrics whose provenance is unresolved, and the unvalidated sensor and decision branches. The component boundaries are intentional: shared storage and a dashboard can connect results operationally, but a shared interface does not demonstrate statistical fusion or better decisions. The architecture becomes an empirical integrated system only after synchronized field records and an evaluation of the complete decision path are available.</p>
''')
parts.append(fig("fig01_smartagro_architecture", "Figure 1. Proposed SmartAgro architecture with evidence status shown by color. The satellite component is an offline-reproduced regional detection-ranking experiment. Disease metrics are reported artifacts without a linked run manifest. Sensor ingestion, multimodal fusion, and farmer-facing decision support are proposed and unevaluated."))

parts.append(h2(5, "Datasets, provenance, and data access"))
parts.append('''
<p><b>PlantVillage image tree.</b> The extracted local color tree contains 38 class folders and 54,305 image files, one fewer than the 54,306 images described in the original PlantVillage study and the included README [1]. The source folder names encode crop and condition, and the available RGB images are controlled leaf photographs. The reported classifier CSVs contain 16 class names, corresponding to the first 16 labels in the checked-in script's sorted class order. Their test supports sum to 3,563. The exact `--data` path and class-filtering operation used to produce these files are not retained, so the current 38-class tree cannot by itself establish which input subset was used. The local README and citation metadata identify the dataset paper but no explicit data license file was found in the extracted repository. Redistribution rights must be confirmed from the source before republishing images.</p>
<p><b>FIRMS and POWER cache.</b> The main fire cache contains 91 CSV request files named by five-day date windows and 54 POWER JSON files, one for each grid point. The run log records `mock: false`, years 2017–2023, a bounding box from 74.0°E to 77.6°E and 28.4°N to 32.6°N, a 0.5° × 0.625° grid, and October–November. The result JSON records 13,176 training, 3,294 validation, and 6,588 test cell-days. The cache supplies enough information to reconstruct these row counts and metrics. However, the run log does not record request URLs, retrieval hashes, a source product version, or response metadata. The exact metric replay verifies computational consistency with these files; it does not independently certify the files' remote provenance or immutability.</p>
<p><b>Other materials.</b> `data/firms_test.csv` is a separate VIIRS-shaped sample dated 1 November 2024; it is not used in the 2017–2023 result replay and has no associated collection manifest. The `results_c_debug` directory contains a one-year run with zero validation and test positives and a fallback temporal split; those metrics are not interpretable and are not used. Two large files named as ZIP archives begin with ZIP signatures but are rejected by Python's ZIP reader and are not used as validated provenance. The extracted image tree and the saved fire cache are the materials used for this audit.</p>
<p><b>Unobserved modalities.</b> There are no farm sensor logs, sensor calibration records, synchronized images and environmental measurements, expert-confirmed disease or stress labels, crop-stage records, action execution records, or independent field outcomes. The absence of these data places a hard limit on claims about environmental monitoring, fusion, and action effectiveness. Table 2 summarizes the source, coverage, intended use, and verification status.</p>
''')
data_rows = [
    ["PlantVillage color tree", "Local extracted `raw/color`; 38 folders, 54,305 images", "Controlled RGB leaf images; 14 crop species in canonical dataset", "Disease classification", "Image count differs by one from the cited 54,306; no local data license file"],
    ["Disease result artifacts", "Two per-class CSVs; two PNG confusion matrices", "16 labels; 3,563 summed test supports", "Reported model comparison", "Internally coherent; invocation, model outputs, predictions, and run JSON absent"],
    ["NASA FIRMS cache", "91 CSV pulls in main cache; VIIRS/S-NPP product path in script", "Oct–Nov in 2017–2023; regional bounding box", "Next-day detection label and lags", "All saved files replay; no response hashes or independent source authentication"],
    ["NASA POWER cache", "54 JSON grid-point files", "Daily T2M, RH2M, WS2M, PRECTOTCORR; grid-cell series", "Historical weather features", "Replayable; time-standard request omitted, so API default applies"],
    ["`firms_test.csv`", "One local CSV, 2024-11-01 sample", "VIIRS-like detection rows", "Test fixture or sample; not main evaluation", "No collection provenance; excluded from metrics"],
    ["Sensor and field records", "Not present", "No timestamps, plot IDs, calibration, or ground-truth labels", "Stress monitoring and fusion", "Not evaluated"],
    ["Existing manuscript", "User-supplied DOCX; framework plus literature review and synthetic examples", "No field trial; simulation values are explicitly synthetic in the draft", "Research framing and proposed design", "A project-specific reference is incomplete; some draft citations are flagged for review"],
]
parts.append(table("Table 2. Dataset and evidence inventory.", ["Source", "Local artifact", "Coverage / contents", "Use", "Verification status"], data_rows, "compact"))

parts.append(h2(6, "Methodology: crop disease classification"))
parts.append('''
<p><b>Task and class mapping.</b> The result CSVs report a 16-class single-label classification task. Each class is represented by a crop–condition folder name; the 16 labels are listed in Table 4 in the order implied by the sorted class folders in the checked-in script. The class support count is the number of test examples per row. Summing those supports gives 3,563. This count is verified against the CSV artifacts, not against a retained list of test image paths. The full local color tree has 38 classes, so the supplied task is a subset and must not be described as the complete PlantVillage benchmark.</p>
<p><b>Declared preprocessing and model code.</b> The checked-in `exp_a_disease.py` collects image paths from immediate class directories, converts files to resized RGB tensors, and defines two models. The small CNN uses three convolution blocks with 32, 64, and 128 filters, max pooling, global average pooling, a 128-unit dense layer, dropout, and a softmax output. MobileNetV2 is initialized with ImageNet weights, its convolutional base is frozen, and a global-average-pooling and softmax head is trained. Both models use horizontal flips, random rotations, brightness adjustment, Adam, and sparse categorical cross-entropy. The script's default settings are 128 × 128 inputs, batch size 32, six epochs, seed 0, and a perceptual-hash Hamming threshold of four. These are properties of the checked-in code. They are not established as the exact settings of the run that produced the CSVs because the corresponding JSON manifest is absent.</p>
<p><b>Split protocol in code.</b> The script computes 64-bit difference hashes and uses a five-chunk search to find candidate pairs within the requested Hamming distance, merges linked images into groups, and forms five stratified group folds. It combines the first three folds for training, uses the fourth for validation, and the fifth for test. A random split is also coded as a deliberately leaky comparator, but no random-split scores are supplied. The result filenames contain `grouped`, which is consistent with the grouped pathway; there is no saved group assignment or split manifest to confirm that this exact procedure was executed on the result input. Thus, the available evidence does not establish that near-duplicate groups were kept separate in the reported test set.</p>
<p><b>Metrics.</b> Accuracy is the fraction of correctly classified test images. Macro-F1 is the unweighted mean of per-class F1 values, so each class contributes equally regardless of its support. Per-class precision, recall, F1, and support are taken from the supplied classification-report CSVs. For the 16 labels, the CSV summary rows reproduce the advertised aggregate metrics to rounding precision. No confidence intervals, repeated-run dispersion, calibration, class-wise error counts, or statistical tests are available. The existing confusion-matrix images are retained as supplied; they do not include numeric cell annotations or a colorbar, and the numeric matrix arrays were not saved.</p>
<p><b>Leakage and domain validity.</b> The grouping script's approximate visual hash is intended to reduce leakage from repeated views, but hash grouping is not equivalent to biological leaf identity or farm-level separation. It can miss transformed near-duplicates or group distinct leaves with similar backgrounds. A group split over the same source collection also does not test acquisition at a new farm. PlantVillage's controlled backgrounds can simplify visual recognition and do not establish field-photo reliability [1]. A separate field-like benchmark such as PlantDoc can provide a transfer test, but its class map and image source differ and require an explicit mapping [2].</p>
''')

parts.append(h2(7, "Methodology: environmental stress monitoring"))
parts.append('''
<p>No empirical environmental-stress model is evaluated in this project. The intended sensor branch is specified here as a study design so that future data collection can be audited against a defined unit of analysis. A candidate record would link a timestamped plant observation to a plot identifier, crop, cultivar where known, planting date or growth stage, sensor device, and independent stress assessment. A sensor value without this context is not a crop-stress label. Weather and soil measurements describe exposures; a field assessment or validated agronomic measurement is required to establish the outcome of interest.</p>
<p>The proposed collection variables and their roles are summarized in Table 8. Air temperature and relative humidity characterize ambient conditions; leaf temperature and leaf-wetness duration are closer to plant exposure; volumetric soil-water content or a calibrated matric-potential sensor describes root-zone water availability; rainfall and solar radiation characterize recent forcing; wind speed supports heat and fire context; electrical conductivity may be useful where salinity is a relevant local stressor. Crop stage and management observations are essential covariates. A future irrigation rule should use crop evapotranspiration and soil-water-balance reasoning rather than a generic sensor threshold [10]. The list is a protocol proposal, not a record of sensors installed or measurements taken.</p>
<p>Each sensor should be calibrated against a traceable reference before deployment, checked during and after the season, and logged with model, firmware, installation depth or height, units, measurement interval, and missing-data flags. Soil moisture requires a soil-specific calibration or a carefully justified correction because texture, bulk density, salinity, and installation affect the reading. Leaf-wetness and canopy-temperature sensors should be positioned and maintained so their measurement represents the plant microclimate rather than a convenient but unrelated location. Sensor drift and clock offset should be quantified rather than silently smoothed away.</p>
<p>Labels should be defined before model fitting. Disease labels require diagnosis by trained agronomists or plant pathologists, with uncertainty and co-infection recorded. Heat and water stress labels should combine crop-stage context with independent physiological or agronomic indicators rather than thresholds selected after viewing model scores. Repeated assessments should use blinded adjudication on a subset. A future model comparison should include image-only, sensor-only, simple rule-based, and fused baselines; report leave-farm-out and forward-season results; and test whether adding a modality improves calibrated decision quality rather than only an aggregate score. Until such paired observations are collected, this component remains unvalidated.</p>
''')

sensor_rows = [
    ["Air temperature", "°C", "Continuous or sub-hourly", "Ambient heat exposure; quality-check against a reference"],
    ["Relative humidity", "%", "Continuous or sub-hourly", "Humidity context; pair with temperature and canopy conditions"],
    ["Leaf / canopy temperature", "°C", "Repeated or continuous", "Plant-level thermal response; record view geometry and sensor type"],
    ["Soil water content or matric potential", "% VWC or kPa", "Sub-hourly to hourly", "Root-zone water status; document depth, soil, and calibration curve"],
    ["Leaf-wetness duration", "minutes per interval", "Sub-hourly", "Potential infection-period feature; sensor placement matters"],
    ["Rainfall and solar radiation", "mm; W/m²", "Hourly or daily summary", "Water input and evaporative demand; preserve source and time basis"],
    ["Wind speed / direction", "m/s; degrees", "Hourly", "Ventilation, canopy drying, and fire-weather context"],
    ["Crop, cultivar, stage, and management", "Categorical / dates", "At planting and each assessment", "Required context for interpreting sensor values and labels"],
]
parts.append(table("Table 8. Proposed environmental observations and collection controls; no such sensor dataset was supplied.", ["Variable", "Units", "Proposed interval", "Purpose and control"], sensor_rows, "compact"))

parts.append(h2(8, "Methodology: satellite fire-risk assessment"))
parts.append('''
<p><b>Geographic and temporal unit.</b> The code defines a rectangular study window from 74.0°E to 77.6°E and 28.4°N to 32.6°N and maps its grid centers to 0.5° latitude × 0.625° longitude cells. There are 54 cells. This bounding box approximates a regional window around parts of Punjab and Haryana; it is not a district boundary or cropland mask. The analysis covers October and November from 2017 through 2023. Daily weather and hotspot data are retained across the calendar to construct antecedent variables, then evaluation rows are restricted to the target season.</p>
<p><b>Target definition.</b> For cell <i>g</i> and date <i>t</i>, the binary label is 1 when at least one VIIRS/Suomi NPP detection with confidence not equal to the low code occurs in cell <i>g</i> on date <i>t + 1</i>; otherwise it is 0. This is an event-observation target from satellite data. There is no cropland mask in the main run and no independent ground-truth record of ignition, crop-residue burning, or fire cause. Multiple detections within one cell-day are collapsed to presence. Persistence of a detection across days can therefore contribute strongly to the score without implying prediction of a new ignition.</p>
<p><b>Predictors and alignment.</b> NASA POWER daily fields are air temperature at 2 m (T2M), relative humidity at 2 m (RH2M), wind speed at 2 m (WS2M), and corrected precipitation (PRECTOTCORR). The script derives a rolling three-day precipitation sum and a dry-day run length. It also includes current-day hotspot presence, previous one-day presence, rolling three- and seven-day counts, day of year, latitude, and longitude. Lags are computed using the preceding calendar history. The script's weather heuristic is exactly the standardized combination z(T2M) − z(RH2M) + z(WS2M); despite its label, it does not directly include a rainfall or fuel-moisture term. Logistic regression uses standardized inputs and a maximum of 2,000 iterations. The histogram gradient-boosted classifier uses maximum depth 3, learning rate 0.08, 150 iterations, and random state 0.</p>
<p>A material temporal-alignment issue remains. The POWER request URL does not set `time-standard=UTC`; the NASA daily API documentation states that local solar time is the default [8]. FIRMS acquisition dates are represented in UTC [7]. The current code joins those date strings directly. A day boundary in local solar time and UTC need not represent the same observation interval. Exact metric replay does not resolve this mismatch. A production analysis should request a common time standard or define a reproducible conversion from the original observation timestamps and then rebuild features and labels.</p>
<p><b>Baselines and split.</b> The persistence baseline scores a cell by whether it had a detection on date <i>t</i>. The climatology baseline scores it by the training-set positive rate for the same day of year. The standardized weather index provides a simple weather-only diagnostic. The year-based split uses 2017–2020 for training, 2021 for validation, and 2022–2023 for test. The observed cells are repeated in each period; the split tests future years at locations represented in training, not geographic transfer to unseen cells. A fallback date-fraction split exists in the code but is not used for the main result.</p>
<p><b>Metrics and thresholds.</b> The result file labels `average_precision_score` as AUPRC. We retain that reported label and clarify that the computation is scikit-learn average precision. The test prevalence is 0.4954, which is an important reference for interpreting average precision; the near-balanced rate does not remove the need to compare with baselines. The model threshold is selected separately on validation scores to target approximately 80% cumulative recall, then held fixed for test reporting. Test precision, recall, missed-event rate, false alarms per 100 negative cell-days, and alert share are calculated at that threshold. Brier scores are provided for climatology, logistic regression, and boosted trees. A 95% interval is estimated with 200 bootstrap resamples of test weeks for average precision. The same weeks are not spatially independent, and no paired significance test between model scores is available.</p>
<p><b>Operational interpretation.</b> A retrospective model using reanalysis or observed daily weather does not reproduce the information available before the event. An operational forecast would require weather forecasts available at prediction time, a declared forecast lead, and an evaluation that respects forecast issuance. Satellite detections may arrive after an overpass and may be delayed or revised. At this grid resolution the prediction is regional and cannot identify which field should receive a warning. Accordingly, the experiment is interpreted as a retrospective ranking of next-day grid-cell detection, not as demonstrated early prevention of agricultural fires.</p>
''')

features_rows = [
    ["Disease image", "One image → one of 16 reported classes", "Grouped train/validation/test is declared in code; output provenance absent", "Accuracy, macro-F1, per-class metrics"],
    ["Fire detection", "Cell-date → non-low-confidence VIIRS detection on next day", "Train ≤2020; validation 2021; test 2022–2023", "Average precision; validation-threshold test metrics"],
    ["Environmental stress", "No empirical target defined in supplied data", "No sensor split or paired examples", "Not available"],
    ["Multimodal support", "No joint image–sensor–outcome records", "No fusion evaluation", "Not available"],
]
parts.append(table("Table 7. Evaluated targets, features, and split definitions.", ["Component", "Unit and target", "Split / status", "Available evaluation"], features_rows, "compact"))

parts.append(h2(9, "Methodology: multimodal fusion and decision support"))
parts.append('''
<p>The proposed decision layer should join records at the farm or plot scale only when the spatial support and time basis are compatible. An image should carry capture time and plot identity. Sensors should carry calibrated values and clock uncertainty. Weather should identify whether it is observed, reanalysis, or forecast data, as well as the issue time and forecast lead. Satellite inputs should retain product, overpass, acquisition time, pixel support, confidence code, and delay. Missing inputs should be represented explicitly rather than imputed without a quality flag.</p>
<p>A valid multimodal experiment would compare at least four systems on the same independently labelled examples: image-only, sensor-only, a simple combined-feature baseline, and the proposed fused model. The test set must be separated by farm or plot and by time according to the intended deployment, so that views of the same plant and repeated sensor episodes cannot cross the evaluation boundary. All models must use identical label definitions and sample inclusion rules. A paired bootstrap over farms or field episodes would estimate uncertainty in metric differences. Calibration, abstention, per-crop sensitivity, and performance under missing modalities should be reported.</p>
<p>Decision support should be evaluated as a policy, not only as a classifier. A notice must state the observation source, geographic scale, time, confidence, and suggested next step. A model score should not become a treatment instruction until agronomists define safe, crop-specific actions and escalation rules. For an irrigation recommendation, the system can log whether a pump or valve ran and whether soil water changed; this does not by itself prove a yield benefit. For fire risk, the system can distinguish regional situational awareness from local fire detection, and should route an observed hotspot for human verification rather than infer cause. Figure 8 shows these missing links and the proposed evidence needed to evaluate them.</p>
''')
parts.append(fig("fig08_proposed_multimodal_fusion", "Figure 8. Proposed multimodal fusion pathway. Timestamp normalization, farm/plot identifiers, calibrated sensors, independent stress labels, and action outcomes are not present in the supplied materials; all integration links remain future work."))

parts.append(h2(10, "Experimental setup and reproducibility"))
parts.append('''
<p>The fire pipeline is implemented in `exp_c_fire_risk.py`. It records Python 3.11.9, pandas 3.0.6, NumPy 1.26.4, `mock: false`, years, bounding box, grid spacing, and season. The result JSON records seed 0, split mode, feature names, split sizes, class counts, and metrics. It does not record scikit-learn version, package lock, source hashes, the map key, API response headers, a source product checksum, or a cropland mask. The MAP_KEY itself is not present in the audit outputs. We reran the saved data through `fetch_firms` and `fetch_power` in cache-only mode, reconstructed the dataset, and called the checked-in evaluation function with the logged 2020/2021 split. Every split count, positive count, prevalence, AUPRC value, and threshold-dependent metric matched the saved JSON exactly. This is an exact computational reproduction from supplied local inputs.</p>
<p>The disease script declares a Python/TensorFlow pipeline and writes a `results_a.json` manifest containing dataset counts, groups, model settings, run time, and metrics. That manifest is absent from the current `results_a` directory. The two CSV files and two PNG images are insufficient to recover the input paths, model weights, training history, class selection, split IDs, predictions, software versions, or hardware. The historical experiment-status files record Python 3.11, TensorFlow 2.16.2, NumPy 1.26.4, protobuf 4.25.9, and datasets 2.19.0 while stating that real metrics were blocked. Their timestamps predate the disease artifacts and the main fire result files. The status record is therefore stale with respect to those later files, but the later disease files still lack provenance.</p>
<p>The status audit also notes that a prior Hugging Face loader path returned a text-only schema and that its `color` configuration was unavailable. This observation explains why that loader could not support training; it does not negate the local extracted image tree now present in the workspace. The result files appear after the status note, and the main fire cache is available for replay. Conversely, the status note is not evidence that a particular disease run occurred. Figure 2 summarizes this evidence workflow. No additional model training or field experiment was performed for this paper.</p>
<p>Reproducibility instructions, package versions actually recorded, cache replay steps, and known omissions are provided in `reproducibility.md`. The fire replay verifies deterministic agreement for the available files and seed; it does not provide a fresh ground-truth validation or resolve UTC/local-solar date alignment. Disease score reproduction would require the missing run manifest or a new, explicitly declared run; the present paper does not silently reconstruct or replace those values.</p>
''')
parts.append(fig("fig02_evidence_workflow", "Figure 2. Evidence workflow. The historical blocked status report predates the later result files; the fire component is exactly replayed from cache, while the disease artifacts remain provisional and the integrated field study remains future work."))

parts.append(h2(11, "Results: crop disease classification"))
parts.append('''
<p>The disease CSV artifacts report an accuracy of 97.614% and macro-F1 of 96.922% for MobileNetV2, and 75.751% and 65.347% for the small CNN. The reported test support is 3,563 for both model summaries. The per-class macro-F1 values average to the CSV's macro row; the accuracy row and total support agree with the reported aggregate. These arithmetic checks establish internal consistency. They do not reproduce the scores from image paths or model predictions because no prediction-level output or class-by-class integer confusion matrix is retained.</p>
<p>Figure 5 displays the two aggregate artifact scores on the same scale. Within this reported subset and under the unverified split provenance, the MobileNetV2 summary is higher on both metrics. This difference is compatible with the value of pretrained representations for a limited-data image task, but it cannot be attributed solely to transfer learning: the models differ in architecture and initialization, and the run record does not establish identical optimization histories, the exact subset, or data order. It does not show that MobileNetV2 is universally better for agricultural diagnosis.</p>
<p>Per-class CSV evidence clarifies where the aggregate gap arises. The small CNN has an F1 of 0 for Apple cedar apple rust (support 55), 0.294 for grape black rot (236), 0.332 for cherry powdery mildew (210), and about 0.455 for both apple scab (126) and corn Cercospora/gray leaf spot (103). MobileNetV2's corresponding reported F1 values are 1.000, 0.959, 0.990, 0.964, and 0.806. Its lowest reported class F1 is for corn Cercospora/gray leaf spot at 0.806; the next-lowest is corn northern leaf blight at 0.904. The orange HLB class has the largest support (665) and an F1 of 0.998 for MobileNetV2 versus 0.922 for the small CNN. These values come directly from the class CSVs and should be read as provisional.</p>
<p>Figures 3 and 4 retain the supplied matrices and add the class-index mapping implied by the sorted-label script and CSV order. The original images contain no colorbar or cell annotations, and the numeric arrays are missing; we therefore do not infer off-diagonal counts from their pixels. Table 4 reports the support and F1 values for every class. The matrix images visually suggest stronger diagonal concentration for the MobileNetV2 artifact, but they do not support a precise account of individual error pairs. The per-class CSVs support class-level F1 comparisons, not claims about which exact disease was mistaken for which other disease.</p>
<p>Because the classifier metrics are not tied to a run manifest, the evidence does not establish the grouping algorithm, absence of duplicate leakage, 16-class selection procedure, model training settings, or held-out sample identities. The 3,563 count is a verified sum of CSV support values, not a verified count from an independent test manifest. Even if the intended grouped split was used, a split based on near-duplicate images from PlantVillage would not estimate performance on field photographs, new farms, or new camera conditions. The most defensible result is thus a documented but provisional within-subset model comparison.</p>
''')

disease_rows = [
    ["MobileNetV2", pct(mobile.loc["accuracy", "precision"]), pct(mobile.loc["macro avg", "f1-score"]), "3,563", "Provisional artifact summary"],
    ["Small CNN", pct(small.loc["accuracy", "precision"]), pct(small.loc["macro avg", "f1-score"]), "3,563", "Provisional artifact summary"],
]
parts.append(table("Table 3. Reported disease-classification aggregate metrics.", ["Model", "Accuracy", "Macro-F1", "Test support", "Verification"], disease_rows, "compact"))
parts.append(fig("fig05_disease_model_comparison", "Figure 5. Reported accuracy and macro-F1 for the 16-class disease subset. Bars reproduce the CSV summary values; they are provisional because the run manifest, predictions, and exact data selection are unavailable."))
parts.append(fig("fig03_mobilenetv2_confusion_matrix", "Figure 3. Supplied MobileNetV2 confusion-matrix image. The 16 class indices map to Table 4. Row/column labels and color scale were absent from the supplied PNG; class counts are not reconstructed."))
parts.append(fig("fig04_smallcnn_confusion_matrix", "Figure 4. Supplied small-CNN confusion-matrix image. It uses the same class order as Figure 3. The source PNG lacked a color scale and numeric cell annotations; exact confusion counts are unavailable."))

class_rows = []
class_names = [x for x in mobile.index if x not in ("accuracy", "macro avg", "weighted avg")]
for i, name in enumerate(class_names):
    disp = name.replace("___", " — ").replace("_", " ")
    class_rows.append([str(i), disp, str(int(mobile.loc[name, "support"])),
                       f"{small.loc[name, 'f1-score']:.3f}", f"{mobile.loc[name, 'f1-score']:.3f}"])
parts.append(table("Table 4. Reported per-class F1 and support. Class names and order follow the CSV artifacts; spellings are retained from the source labels.",
                    ["Index", "Class label", "Support", "Small-CNN F1", "MobileNetV2 F1"], class_rows, "class-table"))

parts.append(h2(12, "Results: satellite fire-risk assessment"))
parts.append('''
<p>The cache replay reconstructs 23,058 cell-day examples across the seven October–November seasons: 13,176 training, 3,294 validation, and 6,588 test rows. The train, validation, and test positive counts are 6,984, 1,805, and 3,264. The test positive rate is 0.495446, matching the saved result. All five AUPRC values and the reported threshold-dependent test metrics match the `results_c.json` values exactly when the local cache is replayed with the checked-in code and seed.</p>
<p>Gradient-boosted trees reach average precision 0.8723, followed by logistic regression at 0.8530. Their absolute difference is 0.0193. The supplied 95% weekly block-bootstrap intervals are 0.801–0.913 for boosted trees and 0.797–0.894 for logistic regression, with substantial overlap. Persistence reaches 0.7409, day-of-season climatology 0.5452, and the weather heuristic 0.4369. The climatology score is modestly above the 0.495 test prevalence reference; the weather-only score is below it. The persistence score shows that recent same-cell detection is a strong predictor of another next-day detection in this constructed target. No paired uncertainty analysis or statistical test establishes a reliable advantage of the boosted model over logistic regression.</p>
<p>Figure 6 presents average precision with the reported test prevalence and the supplied week-bootstrap intervals. The intervals summarize temporal resampling of weeks but do not account for the spatial dependence among the 54 adjacent grid cells. They are descriptive uncertainty estimates for the chosen test period, not a confidence interval for future farms or all of Punjab and Haryana. Because the grid sites repeat between train and test and latitude/longitude are predictors, the model can use stable spatial patterns. No spatial holdout or leave-region-out evaluation is provided.</p>
<p>The operational threshold metrics offer another view. The validation threshold targets approximately 80% recall; test recall is 80.1% for persistence, 79.4% for climatology, 63.0% for the weather index, 70.0% for logistic regression, and 73.2% for boosted trees. The boosted model alerts 43.3% of test cell-days, with 83.7% precision and 14.0 false alarms per 100 negative cell-days. Logistic regression alerts 40.8%, with 85.0% precision and 12.1 false alarms per 100 negative days. Persistence alerts 49.5% and has 80.2% precision; climatology alerts 77.0% and has 51.0% precision. Figure 7 separates alert coverage/recall from false-alarm burden so unlike quantities are not overlaid on one scale.</p>
<p>For the two fitted probabilistic models, Brier scores are 0.1346 for logistic regression and 0.1285 for boosted trees; the climatology score is 0.2500. These values are available in the result JSON, but no reliability diagram or calibration analysis was supplied. They should not be interpreted as evidence of calibrated operational probabilities. The observed weather predictors were available retrospectively. The experiment does not show performance using a forecast issued before the target day.</p>
<p>Model performance cannot be attributed to weather without ablation. The full models include current and lagged hotspot detections, day of year, coordinates, and weather-derived features. The high persistence score and fixed geographic grid make temporal autocorrelation, seasonal patterns, and location plausible explanations for part of the skill. An experiment removing weather, removing hotspot lags, and removing coordinates is needed before claiming a weather contribution. The target also measures repeated satellite detection, not ignition, fire size, crop loss, or residue burning. Thus, the result supports a retrospective regional detection-ranking comparison, not proof of preventable fire risk.</p>
''')

fire_order = ["persistence (fire yesterday)", "climatology (day-of-season rate)",
              "weather index (hot, dry, windy)", "logistic regression", "gradient-boosted trees"]
fire_labels = ["Persistence", "Day-of-season climatology", "Weather index", "Logistic regression", "Gradient-boosted trees"]
ap_rows = []
for key, label in zip(fire_order, fire_labels):
    m = models[key]
    ap_rows.append([label, f"{m['AUPRC']:.4f}", f"{m['AUPRC_CI95_block_bootstrap'][0]:.4f}–{m['AUPRC_CI95_block_bootstrap'][1]:.4f}",
                    "Not reported" if m["brier"] is None else f"{m['brier']:.4f}"])
parts.append(table("Table 5. Cache-reproduced test ranking and probabilistic scores.",
                    ["Model", "Average precision", "95% weekly block-bootstrap interval", "Brier score"], ap_rows, "compact"))
parts.append(fig("fig06_fire_auprc", "Figure 6. Cache-reproduced average precision for next-day non-low-confidence VIIRS detection. The dashed line marks test positive prevalence (0.4954), shown as a reference. Error bars are the supplied 95% block-bootstrap intervals over test weeks."))

oper_rows = []
for key, label in zip(fire_order, fire_labels):
    m = models[key]
    oper_rows.append([label, f"{m['precision']*100:.2f}%", f"{m['recall']*100:.2f}%",
                      f"{m['missed_event_rate']*100:.2f}%", f"{m['alert_share_pct']:.2f}%",
                      f"{m['false_alarms_per_100_negative_cell_days']:.2f}",
                      "Not reported" if m["brier"] is None else f"{m['brier']:.4f}"])
parts.append(table("Table 6. Test-set operational metrics at model-specific validation thresholds targeting approximately 80% recall.",
                    ["Model", "Precision", "Recall", "Missed-event rate", "Alert share", "False alarms / 100 negative cell-days", "Brier"], oper_rows, "compact"))
parts.append(fig("fig07_fire_operational_tradeoffs", "Figure 7. Test-set trade-offs at thresholds chosen separately on validation data to target approximately 80% recall. The left panel shows alert share against test recall. The right panel shows false alarms per 100 negative cell-days; annotations give test precision."))

parts.append(h2(13, "Environmental monitoring and fusion status"))
parts.append('''
<p>No empirical results are available for environmental stress monitoring, synchronized image–sensor fusion, or preventive actions. This is a positive finding of the evidence audit: it prevents the proposed framework from being mistaken for a field-tested system. The status report, research plan, and existing manuscript all describe the missing paired data, and no contrary sensor dataset or field log was found in the workspace.</p>
<p>The sensor branch has not produced estimates of heat stress, water stress, leaf wetness, or disease severity. The fire model's NASA POWER predictors are gridded historical meteorology, not on-farm IoT readings and not forecasts. No model combines these weather fields with disease images. The disease artifacts include only class predictions summarized in CSVs; no timestamps, plots, weather, or sensor values accompany them. No action recommendation was delivered and independently confirmed, so no closed-loop response can be measured.</p>
<p>A valid fusion dataset must provide synchronized timestamps, farm and plot identifiers, calibrated sensor measurements, plant images, crop and growth-stage metadata, independently confirmed stress or disease labels, management history, and a locked external test set. The records must identify whether each environmental field is a direct sensor value, a derived feature, a gridded analysis, a synthetic stand-in, or a model output. The current study contains no synthetic results in its reported component tables; synthetic examples in the earlier draft are kept out of the empirical results because they demonstrate code logic or assumptions, not agricultural performance.</p>
''')

parts.append(h2(14, "Discussion"))
parts.append('''
<p>The evidence supports two different levels of conclusion. The disease artifacts show a large reported gap between a MobileNetV2 model and a small CNN on a 16-class test subset, and the class CSVs indicate that the difference is not uniform across classes. The fire experiment is more reproducible at the computation level: its saved input cache and checked-in code generate the same row counts, class prevalence, ranking scores, and threshold metrics as the result JSON. Neither component establishes an integrated farm-monitoring capability. Preserving those evidence levels is central to the paper's interpretation.</p>
<p><b>What the disease comparison does and does not mean.</b> A high score on a controlled-image subset is evidence that the supplied result artifact reports separable visual classes under some held-out condition. The large difference between the two model summaries is compatible with the use of pretrained visual features, especially when the task dataset is limited. Yet the exact data subset and execution are not documented, and there is no repeated seed, confidence interval, or independent test. The class-level pattern also matters: the small CNN fails on the reported Apple cedar rust support of 55 while performing well on common rust and healthy corn; MobileNetV2 remains weakest on a corn leaf-spot class. A macro-F1 near 0.97 does not mean every disease is equally dependable, and the reported metrics should not be used to guide treatment.</p>
<p>The comparison also cannot answer whether near-duplicate leakage was prevented. Grouping is described in the code and suggested by filenames, but the output manifest is absent. If grouped splitting was used, it would be a stronger internal validation design than a random image split for repeated leaf views. It would still not provide external validity to field photographs. The original PlantVillage work itself reported a pronounced decline on images from outside its controlled collection [1]. A future paper should therefore show the full 38-class benchmark under a documented group split, then report a separate field-like test with its own class mapping and uncertainty.</p>
<p><b>What the fire replay establishes.</b> Exact recomputation resolves the apparent conflict between the earlier status note and the later fire result files: the historical note predates them, while the saved cache and pipeline support reproducibility of the reported numerical calculation. This result is meaningful as a regional retrospective ranking experiment. At the same time, agreement between result JSON and a local cache does not establish the authenticity, version, or completeness of the cache. A request manifest, URL without credentials, observation timestamps, file hashes, and data-product version should be recorded in future runs.</p>
<p>The high persistence score points to temporal continuity in the label process. Since a positive is any qualifying satellite detection the next day, an ongoing burn can generate positives on consecutive dates. The model may rank cells well because the event is already visible at time <i>t</i>. That can be useful for monitoring continued activity, but it differs from predicting a new ignition early enough to prevent one. The performance of logistic regression and boosted trees also reflects a feature set containing location, day of year, and recent hotspot counts. No ablation allows attribution to weather, and no spatial holdout tests transfer to unobserved regions.</p>
<p>The ranking results do not remove operational trade-offs. The boosted model has higher average precision than logistic regression, but their supplied block-bootstrap intervals overlap and the test-threshold metrics give logistic regression somewhat higher precision and fewer false alarms per negative cell-day, while the boosted model has higher recall. A deployment decision would depend on how often an alert can be reviewed, the cost of a missed event, and how many false alarms an extension team can handle. Those costs were not elicited, so a single “best model” is premature. AUPRC ranks scores over all thresholds; it does not select a policy.</p>
<p>The fire label also carries remote-sensing measurement error. VIIRS has a finer nominal footprint than MODIS, but a hotspot remains an observation of thermal activity and can be obscured by cloud or missed between overpasses [6], [7]. The project does not use a cropland mask, ground survey, or cause label. The result cannot be translated into the probability that a farmer's residue-burning activity will cause a fire, nor into avoided losses. Additionally, the time-standard mismatch between the POWER daily fields and FIRMS date labels could shift feature alignment around a day boundary. This issue must be resolved before treating the exact replay as a fully validated data pipeline.</p>
<p><b>Integration is an open empirical question.</b> A dashboard can present a disease result, sensor reading, and regional hotspot map without learning a joint representation. That is a useful software integration but not sensor-image fusion. Fusion should be assessed with paired examples and a fair set of unimodal baselines. Preventive decision support should then be evaluated as a policy, including whether the advice is technically safe, delivered in time, understood, acted upon, and followed by a measurable response. Until then, SmartAgro is a framework proposal with two separate component evidence sets, not a validated end-to-end system.</p>
''')

parts.append(h2(15, "Practical implications"))
parts.append('''
<p>For farmers and extension workers, the current outputs are not ready to support operational diagnosis or fire prevention. The disease values are provisional, and the satellite model predicts a regional cell-day detection outcome rather than a field event. The useful practical contribution at this stage is an interface and data-governance specification: each alert should identify whether it is a model score, sensor measurement, remotely sensed hotspot, or verified observation, and should state its time, location, and spatial resolution.</p>
<p>For an agronomist reviewing a leaf image, a calibrated top-k list with an explicit “uncertain—seek diagnosis” option would be safer than a single disease label. A crop-specific recommendation should include the basis for the advice and should follow locally approved extension guidance. Environmental alerts should account for crop stage and soil conditions; generalized threshold rules can be misleading when soils, cultivars, planting dates, and management differ. This study contributes no treatment thresholds or product recommendations.</p>
<p>For extension services and agricultural decision-makers, a satellite-based map may support regional situational awareness when labeled as a hotspot product. Its grid cell spans an area far larger than a field, so an alert should trigger verification or coordination rather than identify a particular farm. A future operational use case must separately evaluate satellite latency, forecast availability, communications, phone coverage, staff workload, language, accessibility, and farmer trust. These deployment considerations are not measured in the current results.</p>
<p>Finally, any system that logs farm images, sensor streams, locations, or management actions should limit collection to a defined purpose and allow farmers to understand and govern data sharing. Images of fields can contain people, buildings, and property details. Consent, retention, access control, and local ownership should be incorporated before field deployment, not added after a model is built.</p>
''')

parts.append(h2(16, "Limitations and threats to validity"))
parts.append('''
<p>Table 9 consolidates the main threats. Several can be fixed by better documentation or analysis of existing local files; others require new data. The most important interpretive limit is that disease and fire evidence are not equivalent: the fire pipeline can be rerun exactly from supplied cached files, but its source integrity and time semantics remain incompletely documented; the disease metrics are supported only by result summaries and images. Neither constitutes field validation.</p>
<p><b>Construct validity.</b> A PlantVillage class label is not a complete clinical description of the plant or a treatment decision. A VIIRS hotspot is not confirmed agricultural fire or residue-burning ground truth. A daily gridded weather field is not a calibrated on-farm sensor. No operational action or crop outcome is observed. These limits constrain the construct each component can claim to measure.</p>
<p><b>Internal validity and leakage.</b> Disease run provenance is absent, so the split, grouping, class selection, and model settings are not confirmed. The fire split is chronological, but the same grid cells appear across years and spatial predictors are available. The exact metric replay confirms arithmetic against those inputs, not independence of the test observations. Bootstrap intervals sample test weeks and preserve some temporal structure, but not spatial dependence across cells.</p>
<p><b>Temporal and data-source validity.</b> NASA POWER defaults to local solar time and the code does not request UTC, while FIRMS dates are UTC [8], [7]. This could shift which daily meteorology is aligned to a hotspot label. Cached records lack checksums and response metadata, and data updates or revised product versions could affect reproduction. Test years are only 2022 and 2023. No independent region or season is evaluated.</p>
<p><b>External validity.</b> The disease artifacts cover 16 classes, while the local image tree contains 38. The imagery is controlled and does not establish performance on field photographs. The fire study uses a rectangular regional grid without a cropland mask. Weather fields are gridded and retrospectively observed. No forecast product, farmer feedback, sensor reliability, connectivity, latency, hardware cost, or seasonal field operation is measured.</p>
<p><b>Statistical uncertainty.</b> Fire AP intervals overlap; no paired difference test or repeated-year evaluation is available. Disease metrics have no interval estimates and no repeated training seeds. The threshold-selection procedure targets validation recall, while the reported test recall differs; this is expected under distribution shift and finite samples, and it means the target is not guaranteed in deployment. Model selection and threshold costs were not preregistered in a separate analysis plan.</p>
''')

threat_rows = [
    ["Disease split and run provenance", "No manifest, predictions, split IDs, or confusion arrays; 16-class selection is not recorded", "Retain per-image predictions and hashes; log exact inputs, groups, folds, packages, and seeds; lock external test"],
    ["Disease domain shift", "Controlled leaf images; no field-photo evaluation", "Add independent field-like images and external farms; report class mapping and acquisition differences"],
    ["VIIRS target validity", "Non-low-confidence hotspot may be fire or another thermal anomaly; no cause/ground truth", "Independent event verification, cropland masks, and clear event-label adjudication"],
    ["Spatial dependence", "Same 54 cells repeat across temporal splits; lat/lon included", "Add spatial block and leave-region-out tests; report interpolation and extrapolation separately"],
    ["Weather time standard", "POWER local-solar daily default joined directly to UTC FIRMS date", "Request UTC or reconstruct aligned windows from timestamps; rerun complete pipeline"],
    ["Weather attribution", "No ablation; model includes event lags, day of year, and location", "Compare lag-only, weather-only, no-location, full, and forecast-driven models"],
    ["No field fusion", "No synchronized images, calibrated sensors, stress labels, or action logs", "Collect paired farm/plot records with independent labels and external evaluation"],
    ["Operational usefulness", "No forecast inputs, latency, alert cost, response, or yield/fire outcome", "Prospective pilot with alert burden, lead time, action confirmation, safety, and cost measures"],
]
parts.append(table("Table 9. Evidence gaps, threats to validity, and mitigation steps.", ["Threat", "Current evidence gap", "Mitigation"], threat_rows, "compact"))

parts.append(h2(17, "Future work and validation roadmap"))
parts.append('''
<p>Future work should proceed through evidence gates. A reproducible benchmark should be completed before field deployment, and component-level validation should precede a claim about fusion or impact. Figure 9 shows the current stopping point and the proposed sequence. Table 10 specifies the minimum data and evaluation deliverables for each stage.</p>
<p><b>Stage 1: reconstruct the disease run.</b> Recover the exact 16-class source path and invocation, identify whether it was a filtered view of the local 38-class tree, and produce a manifest containing image paths or stable hashes, class counts, duplicate-group IDs, fold assignments, train/validation/test counts, package versions, hardware, random seeds, training logs, model weights, and prediction probabilities. Recompute accuracy, macro-F1, class metrics, and numeric confusion matrices from prediction rows. If the original inputs or model are unavailable, label the archived values historical and run a new benchmark as a new experiment rather than silently substituting it. A full 38-class result and the 16-class result should be reported separately.</p>
<p><b>Stage 2: test image-domain transfer.</b> Establish the split at a plant or image-group level and preserve a final locked test set. Compare model families under a declared tuning budget and repeat training over multiple seeds. Evaluate field-like images separately from controlled images, documenting class alignment, ambiguous labels, image quality, crop, and collection source. Report per-class precision and recall, macro-F1, calibration, abstention behavior, and confidence intervals resampled at the plant or site level. The test should include new sites or seasons if the intended use is field diagnosis.</p>
<p><b>Stage 3: rebuild the fire evaluation with time-aligned inputs.</b> Retain raw request URLs without secrets, data product and collection versions, retrieval timestamps, response headers, file checksums, and data-use metadata. Explicitly align NASA POWER daily time basis with FIRMS acquisition timestamps, then reconstruct all target and lag features. Add a cropland mask or independently verified event labels if the research question concerns agricultural burning. Use a forward temporal test and a spatial block or leave-region-out test. Compare persistence, seasonal climatology, weather-only, lag-only, location-only, and combined models; evaluate a forecast-driven variant using forecasts available at issue time. Report paired block-bootstrap differences, calibration, threshold rules, alert share, false alarms, missed detections, and lead time.</p>
<p><b>Stage 4: collect paired sensor and image observations.</b> Select farms and plots with local agronomists and extension partners. Install calibrated sensors with documented depth, height, units, placement, and maintenance. Capture leaf images and sensor readings with synchronized clocks and plot IDs. Record crop, cultivar where known, growth stage, irrigation, pesticide and other management events, and independently adjudicated disease or stress outcomes. Maintain a separation between data used for training and the farm-season data reserved for external testing. A pilot should estimate data volume and measurement reliability before a statistical power analysis fixes the confirmatory sample size.</p>
<p><b>Stage 5: evaluate decision support prospectively.</b> Compare the full system with current practice or a transparent rule-based workflow. Predefine alert recipients, review burden, escalation rules, safe recommendations, and the threshold used for each risk. Log whether an alert was received, understood, accepted, or acted upon; confirm actions using sensor or operator records; and record outcomes independently. Evaluate utility and harm, including false alarms, missed events, time to response, crop outcomes, cost, connectivity, privacy, and farmer experience. Only this stage can address whether SmartAgro changes decisions or reduces loss.</p>
''')
road_rows = [
    ["1 · Disease provenance", "Run manifest, predictions, exact subset and split", "Recomputed aggregate and per-class scores; grouped and locked test", "Verified benchmark"],
    ["2 · Field transfer", "Independent field-like images with mapped labels", "External farm/season evaluation, calibration and abstention", "Transfer evidence"],
    ["3 · Fire validity", "Time-aligned VIIRS/POWER or forecast inputs; event verification", "Spatial + temporal splits, ablation, paired uncertainty and alert cost", "Regional operational evidence"],
    ["4 · Sensor fusion", "Calibrated sensor streams paired to images and labels", "Image-only, sensor-only, fused, leave-farm-out baselines", "Fusion evidence"],
    ["5 · Decision impact", "Prospective alerts, action logs, independent outcomes", "Comparison with current practice; safety, cost and human factors", "Impact evidence"],
]
parts.append(table("Table 10. Staged validation roadmap and evidence gates.", ["Stage", "Required data", "Evaluation", "Gate"], road_rows, "compact"))
parts.append(fig("fig09_evidence_roadmap", "Figure 9. Evidence and validation roadmap. Completed audit and exact offline fire replay are separated from the unverified disease outputs, absent sensor/fusion evaluation, and future prospective work."))

parts.append(h2(18, "Conclusion"))
parts.append('''
<p>This paper reframes SmartAgro as an evidence-audited research program rather than a field-tested integrated system. The 16-class disease CSVs report strong MobileNetV2 performance and a substantial gap over a small CNN, with 3,563 supported test examples; their arithmetic is coherent, but execution provenance is insufficient for verified benchmark claims. The regional fire pipeline is computationally reproducible from the supplied FIRMS and POWER cache, and its gradient-boosted and logistic models rank next-day satellite detections above the reported baselines. The target is repeated regional hotspot detection, the score intervals overlap, the POWER/FIRMS date alignment requires correction, and the analysis does not establish crop-residue burning or field-level risk.</p>
<p>No empirical evidence supports environmental stress classification, synchronized image–sensor fusion, preventive action verification, or reduced agricultural loss. The defensible contribution is a transparent component assessment, exact replay of the supplied fire metrics, careful qualification of the disease artifacts, and a staged path toward independent field validation. SmartAgro's integrated value remains a testable hypothesis.</p>
''')

parts.append('<div class="references-break"></div><h2>References</h2>')
references = [
    '[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” <i>Frontiers in Plant Science</i>, vol. 7, article 1419, 2016. doi: <a href="https://doi.org/10.3389/fpls.2016.01419">10.3389/fpls.2016.01419</a>.',
    '[2] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, “PlantDoc: A dataset for visual plant disease detection,” in <i>Proc. 7th ACM IKDD CoDS and 25th COMAD</i>, 2020, pp. 249–253. doi: <a href="https://doi.org/10.1145/3371158.3371196">10.1145/3371158.3371196</a>.',
    '[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in <i>Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition</i>, 2018, pp. 4510–4520. doi: <a href="https://doi.org/10.1109/CVPR.2018.00474">10.1109/CVPR.2018.00474</a>.',
    '[4] D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” <i>Ecography</i>, vol. 40, no. 8, pp. 913–929, 2017. doi: <a href="https://doi.org/10.1111/ecog.02881">10.1111/ecog.02881</a>.',
    '[5] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” <i>PLoS ONE</i>, vol. 10, no. 3, e0118432, 2015. doi: <a href="https://doi.org/10.1371/journal.pone.0118432">10.1371/journal.pone.0118432</a>.',
    '[6] W. Schroeder, P. Oliva, L. Giglio, and I. Csiszar, “The new VIIRS 375 m active fire detection data product: Algorithm description and initial assessment,” <i>Remote Sensing of Environment</i>, vol. 143, pp. 85–96, 2014. doi: <a href="https://doi.org/10.1016/j.rse.2013.12.008">10.1016/j.rse.2013.12.008</a>.',
    '[7] NASA LANCE FIRMS, “Fire alerts: Important notes,” official product guidance. Available: <a href="https://firms.modaps.eosdis.nasa.gov/alerts/notes.php">https://firms.modaps.eosdis.nasa.gov/alerts/notes.php</a>. Accessed 22 October 2026.',
    '[8] NASA POWER, “Daily API documentation,” official service documentation. Available: <a href="https://power.larc.nasa.gov/docs/services/api/temporal/daily/">https://power.larc.nasa.gov/docs/services/api/temporal/daily/</a>. Accessed 22 October 2026.',
    '[9] NASA POWER, “Data sources,” official methodology documentation. Available: <a href="https://power.larc.nasa.gov/docs/methodology/data/sources/">https://power.larc.nasa.gov/docs/methodology/data/sources/</a>. Accessed 22 October 2026.',
    '[10] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, <i>Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements</i>, FAO Irrigation and Drainage Paper 56. Rome: Food and Agriculture Organization of the United Nations, 1998. Available: <a href="https://www.fao.org/4/X0490E/X0490E00.htm">https://www.fao.org/4/X0490E/X0490E00.htm</a>.',
]
for r in references:
    parts.append(f'<p class="reference">{r}</p>')

parts.append('<div class="appendix-break"></div><h2>Appendix A. Reproducibility and metric definitions</h2>')
parts.append('''
<p><b>Disease metrics.</b> For each class, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is the harmonic mean of precision and recall, with undefined denominators set to zero in the checked-in script. Accuracy is correct predictions divided by test support. Macro-F1 is the arithmetic mean of class F1 values. The current output CSV permits checking summary arithmetic and support but not recomputing class counts from raw predictions.</p>
<p><b>Fire metric replay.</b> The cache-only reproduction reads each existing `firms_YYYYMMDD.csv` file and each existing `power_lat_lon.json` file under `outputs/results_c/cache`, calls the checked-in `build` and `evaluate` functions for train-end 2020, validation year 2021, and seed 0, and compares the resulting JSON with `outputs/results_c/results_c.json`. The source code does not require a network request when all expected cache files are present. The reproduction confirmed 91/91 expected FIRMS pulls and 54/54 POWER points before executing the replay. The exact scores are reported in the body; no external dataset was fetched for this paper.</p>
<p><b>Interpretation of status.</b> “Reproduced” refers to exact recomputation from supplied local data and code. It does not mean independently verified ground truth, data integrity at download, or operational validity. “Provisional” refers to values present in output artifacts whose original model execution cannot be independently reconstructed.</p>
''')

STYLE = '''
@page { size: A4; margin: 18mm 18mm 17mm 18mm;
  @bottom-right { content: "Page " counter(page); font-family: Arial, sans-serif; font-size: 8pt; color: #56636D; }
}
body { font-family: "Times New Roman", serif; font-size: 11pt; line-height: 1.16; color: #202c36; }
h1 { font-size: 21pt; line-height: 1.12; color: #18324b; margin: 8pt 0 13pt; }
h2 { font-size: 14pt; line-height: 1.15; color: #18324b; margin: 18pt 0 7pt; page-break-after: avoid; }
h2.abstracthead { margin-top: 15pt; }
p { margin: 0 0 7pt; text-align: justify; orphans: 2; widows: 2; }
.titlepage { min-height: 235mm; page-break-after: always; }
.eyebrow { text-transform: uppercase; letter-spacing: 1.1pt; color: #27877b; font: bold 9pt Arial, sans-serif; margin: 18pt 0 12pt; }
.authors { font: bold 11pt Arial, sans-serif; margin: 12pt 0 3pt; }
.affiliation { font: 9pt Arial, sans-serif; color: #56636d; margin-bottom: 16pt; }
.status { background: #eef3f5; border-left: 4px solid #d89b2b; padding: 9pt 11pt; font: 9.2pt Arial, sans-serif; line-height: 1.3; margin: 12pt 0; }
.abstract { font-size: 10.5pt; line-height: 1.2; }
.keywords, .counting { font: 9pt Arial, sans-serif; color: #43515d; margin-top: 9pt; }
figure { margin: 9pt auto 12pt; width: 100%; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption { font: 8.5pt Arial, sans-serif; text-align: justify; color: #344653; line-height: 1.22; margin: 5pt 5pt 0; }
.table-wrap { margin: 9pt 0 13pt; page-break-inside: avoid; }
.tablecap { font: 8.8pt Arial, sans-serif; margin: 0 0 4pt; color: #18324b; }
table { border-collapse: collapse; width: 100%; font: 8.1pt Arial, sans-serif; line-height: 1.15; }
th { background: #18324b; color: white; text-align: left; font-weight: bold; padding: 5px 5px; border: 0.5pt solid #18324b; }
td { vertical-align: top; padding: 4px 5px; border: 0.5pt solid #cbd5dc; }
tbody tr:nth-child(even) { background: #f4f7f8; }
.class-table { font-size: 7.7pt; }
.class-table th, .class-table td { padding: 3px 4px; }
.reference { font-size: 9pt; line-height: 1.18; padding-left: 22pt; text-indent: -22pt; margin-bottom: 7pt; }
.references-break { page-break-before: always; }
.appendix-break { page-break-before: always; }
a { color: #245f83; text-decoration: underline; }
'''

html = '<!doctype html><html><head><meta charset="utf-8"><title>SmartAgro Research Paper</title><style>' + STYLE + '</style></head><body>' + "\n".join(parts) + '</body></html>'
html_path = OUT / "SmartAgro_Research_Paper.html"
html_path.write_text(html, encoding="utf-8")


AUDIT = """# Evidence audit

Audit date: 22 October 2026 (Asia/Kolkata). This audit distinguishes evidence from instructions and proposed design claims. The attached manuscript and project files are source material; their claims were not treated as instructions to repeat unsupported statements.

## Executive finding

The earlier experiment-status report is stale relative to later result artifacts. Fire results in `outputs/results_c/results_c.json` reproduce exactly from the supplied cache and checked-in code. The disease CSVs report internally coherent metrics for a 16-class subset with 3,563 test supports, but the run cannot be authenticated from the available files. Environmental stress sensing, synchronized fusion, preventive action verification, and field impact remain unevaluated.

## File and artifact inventory

| Material | Contents and role | Audit result |
|---|---|---|
| User-supplied SmartAgro DOCX | Prior framework paper, literature review, evaluation design, and synthetic examples | The draft states that it is a proposal/methodology paper and that its simulation uses synthetic data. It contains placeholders (including incomplete reference [23]) and bibliographic items flagged for verification. Its simulated values are not treated as empirical results. |
| `research_plan_and_inventory.md` | Research stages, data inventory, and planned experiments | Useful for proposed design; not execution evidence. |
| `exp_a_disease.py` | PlantVillage collection, dHash grouping, two CNNs, group/random splits, metrics, output writing | Defines intended method and default settings. It would write `results_a.json`, which is absent from the current results folder. |
| `exp_c_fire_risk.py` | FIRMS/POWER acquisition, grid construction, lags, models, time split and metrics | Source code matches the main fire result schema. Contains an observed-weather proxy, no cropland mask in the main run, and no `time-standard=UTC` request. |
| `src/smartagro/data.py`, `metrics.py` | Image grouping and classification-metric utilities | Utility code only; not evidence of a completed benchmark. |
| `tests/test_data_and_metrics.py` | Unit tests for grouping, leakage check, metrics, and file collection | These test helper behavior; they are not model or field experiments. No test suite was run for this audit. |
| Local PlantVillage `raw/color` tree | 38 class folders, 54,305 image files | Counted from the extracted tree. This is one fewer than the 54,306 reported by Mohanty et al. The local repository copy has no explicit data-license file. |
| `data/plantvillage_data.zip`, `plantvillage_git.zip` | Large files named as ZIP archives | Both begin with ZIP signatures but Python's ZIP reader rejects them; they were not used as validated inputs. |
| `outputs/results_a/*.csv` | Two per-class reports | Each covers 16 classes; support sums to 3,563. Aggregate accuracy and macro-F1 match the summary rows. Values are provisional because run provenance and predictions are absent. |
| `outputs/results_a/cm_grouped__*.png` | Two 16×16 matrix plots | Actual supplied images retained. They have numeric indices only, no colorbar or numeric annotations. Row/column labels are mapped using the checked-in sorted-class script and CSV order; off-diagonal counts cannot be recovered reliably. |
| `outputs/results_c/results_c.json` and `run_log.json` | Main fire scores, split, row counts, feature names, run metadata | `mock=false`, years 2017–2023, bounding box/grid/season recorded. No request hashes, API-response metadata, scikit-learn version, or cropland mask recorded. |
| `outputs/results_c/cache` | 91 FIRMS CSV pulls and 54 POWER JSON grid-point files | All expected files for the main run are present and parse through the checked-in pipeline. Cache replay exactly matches every saved headline and threshold metric. File integrity at remote acquisition is not independently established. |
| `outputs/results_c_debug/*` | One-year debug output | Validation and test have zero positives and the split uses fallback temporal fractions. These scores are not interpretable and are excluded. |
| `data/firms_test.csv` | Separate VIIRS-shaped 2024-11-01 CSV sample | Not used in the 2017–2023 replay; provenance is not documented. |
| Status MD/JSON | Earlier claim that disease and fire execution were blocked | Their filesystem timestamps predate the later output artifacts. They are stale as a description of the current workspace. The Hugging Face text-only finding does not account for the local extracted image tree. |

## Numerical audit

### Disease classification

| Model | Accuracy | Macro-F1 | Test support |
|---|---:|---:|---:|
| MobileNetV2 | 97.6144% | 96.9219% | 3,563 |
| Small CNN | 75.7508% | 65.3472% | 3,563 |

The 16 per-class support values sum to 3,563. Macro-F1 values recompute from per-class F1 to the CSV's macro row. The CSVs do not include prediction rows, integer confusion matrices, image identifiers, or split assignments. The PNGs visually encode matrices but lack a scale and are not a reliable source for numeric reconstruction. The files are consistent with a 16-class grouped-split output but do not prove that the checked-in split procedure was used. The full local color tree contains 38 classes; the exact input filter/path is not logged.

### Satellite fire-risk assessment

Cache-only replay first confirmed 91/91 expected FIRMS pulls and 54/54 POWER points. The reconstructed splits are 13,176 train, 3,294 validation, and 6,588 test rows; positives are 6,984, 1,805, and 3,264; test prevalence is 0.4954463. Every saved average-precision, confidence-interval, threshold, false-alarm, alert-share, and Brier value matched the re-run exactly.

The target is a non-low-confidence VIIRS/S-NPP detection in a grid cell on the next date. It is not a confirmed agricultural fire or crop-residue-burning label. The result set uses observed/reanalysis weather, recent detections, day of year, and location; no ablation attributes skill to weather. Test cells are known grid locations, and spatial transfer is not evaluated. POWER daily API default is local solar time, whereas FIRMS dates are UTC; the pipeline joins date strings without an explicit time-basis correction. This remains a material alignment risk.

## Contradiction resolution

The status files say no real metrics are valid. Later timestamps for the main fire JSON/log and disease CSV/PNG outputs show that those files were created after the status report. The main fire result is computationally substantiated by exact cache replay, so the old statement is stale for that experiment. The disease result artifacts likewise postdate the status report, but their provenance is insufficient for validation; they remain reported-but-unverified rather than being promoted to verified results. The debug fire run is not used because its validation and test sets contain no positives.

## Evaluated versus proposed

| Capability | Status |
|---|---|
| 16-class disease score comparison | Reported; arithmetic verified; execution provenance unverified |
| 38-class PlantVillage image inventory | Local files counted; not benchmarked in this paper |
| Regional next-day VIIRS detection ranking | Reproduced from local cache and checked-in code; source-response integrity and time alignment limitations remain |
| Environmental stress estimation from IoT sensors | No supplied data; not evaluated |
| Image–sensor fusion | No paired observations; not evaluated |
| Decision rules / farmer alerts | Proposed only |
| Action verification or loss prevention | No action logs or outcomes; not evaluated |

## Unresolved evidence needs

1. Disease invocation, filtered dataset root, class selection rationale, run JSON, split/group IDs, image hashes, model checkpoints, and per-image predictions.
2. Numeric disease confusion matrices and repeat-seed uncertainty.
3. Explicit PlantVillage data license and the source/condition of the one-image discrepancy.
4. FIRMS response hashes, product version, request metadata, and independent confirmation of cached source provenance.
5. Corrected common time basis for POWER and FIRMS, followed by a fresh metric run.
6. Cropland/event ground truth, spatial holdout, weather/hotspot ablations, and forecast-issued inputs.
7. Calibrated field sensors, synchronized images, crop-stage metadata, expert labels, farm-separated tests, and action/outcome records.
"""
(OUT / "evidence_audit.md").write_text(AUDIT, encoding="utf-8")


REFERENCES_MD = """# References verified for the revised manuscript

Bibliographic details below were checked against the publisher, official data documentation, or the source repository. The citations used in the paper are numbered IEEE-style. Access dates for live documentation: 22 October 2026.

1. S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” *Frontiers in Plant Science*, vol. 7, article 1419, 2016. DOI: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419). Publisher record: [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full). Supports the canonical 54,306-image controlled PlantVillage description and reports the field-source performance limitation.
2. D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, “PlantDoc: A Dataset for Visual Plant Disease Detection,” in *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 2020, pp. 249–253. DOI: [10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196). The author-maintained [dataset repository](https://github.com/pratikkayal/PlantDoc-Dataset) supplies dataset counts and links to the ACM record.
3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018, pp. 4510–4520. DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474). Official [CVPR Open Access record](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html).
4. D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, vol. 40, no. 8, pp. 913–929, 2017. DOI: [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881). Publisher record: [Wiley / Ecography](https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881).
5. T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLoS ONE*, vol. 10, no. 3, e0118432, 2015. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).
6. W. Schroeder, P. Oliva, L. Giglio, and I. Csiszar, “The new VIIRS 375 m active fire detection data product: Algorithm description and initial assessment,” *Remote Sensing of Environment*, vol. 143, pp. 85–96, 2014. DOI: [10.1016/j.rse.2013.12.008](https://doi.org/10.1016/j.rse.2013.12.008). NASA-hosted article copy: [NASA Earthdata](https://www.earthdata.nasa.gov/sites/default/files/imported/Schroeder_et_al_2014b_RSE.pdf).
7. NASA LANCE FIRMS, “[Fire alerts: Important notes](https://firms.modaps.eosdis.nasa.gov/alerts/notes.php),” official FIRMS product guidance. Accessed 22 October 2026. Used for hotspot semantics, confidence cautions, cloud/overpass omissions, and spatial-scale qualifications.
8. NASA POWER, “[Daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/),” official service documentation. Accessed 22 October 2026. Used for daily service behavior and the default local-solar-time setting.
9. NASA POWER, “[Data sources](https://power.larc.nasa.gov/docs/methodology/data/sources/),” official methodology documentation. Accessed 22 October 2026. Used for meteorological source grid and MERRA-2/GEOS provenance.
10. R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*, FAO Irrigation and Drainage Paper 56. Rome: FAO, 1998. Official [FAO text](https://www.fao.org/4/X0490E/X0490E00.htm). Included only to ground the proposed water-stress collection and agronomic validation discussion; no FAO-derived threshold is used as an experimental result.

## Bibliographic items not carried forward

The prior DOCX includes an incomplete project-specific reference [23], a 2026 review entry with missing authors, and several items that its own verification appendix flags for checking. Those entries are not cited in the revised paper. The prior paper's claims about its synthetic examples are not treated as empirical findings. No citation is used in the revised manuscript unless it appears in the numbered list above.
"""
(OUT / "references_verified.md").write_text(REFERENCES_MD, encoding="utf-8")


REPRO = """# Reproducibility record

Audit date: 22 October 2026 (Asia/Kolkata).

## Data and code used

- Disease: `outputs/results_a/per_class_grouped__mobilenetv2.csv`, `per_class_grouped__small_cnn.csv`, and the two supplied `cm_grouped__*.png` files. The local input image tree is `data/plantvillage/PlantVillage-Dataset-master/raw/color` (38 folders, 54,305 files). The exact filter and invocation for the outputs are not recorded.
- Fire: `outputs/results_c/cache/firms_*.csv` (91 files) and `power_*.json` (54 files), `outputs/results_c/results_c.json`, `run_log.json`, and `exp_c_fire_risk.py`.
- Software versions recorded by the main fire run: Python 3.11.9, pandas 3.0.6, NumPy 1.26.4. The fire log omits scikit-learn and dependency-lock versions. A prior environment status records TensorFlow 2.16.2, protobuf 4.25.9, and datasets 2.19.0, but does not connect them to the disease artifacts.

## Fire replay

The exact offline replay performed for this manuscript:

1. Check for all 91 expected FIRMS date-pull CSVs and all 54 POWER grid-point JSON files in `outputs/results_c/cache`.
2. Load them with `exp_c_fire_risk.fetch_firms` and `fetch_power`; all expected files exist, so the functions use local files.
3. Build features and target using `exp_c_fire_risk.build`.
4. Evaluate with `exp_c_fire_risk.evaluate(df, train_end=2020, val_year=2021, outdir=<temporary directory>, seed=0)`.
5. Compare rows, positive counts, prevalence, average precision, bootstrap intervals, test threshold metrics, and Brier scores with `outputs/results_c/results_c.json`.

The replay matched the stored JSON exactly. It is a local computational reproduction, not an independent validation of target labels or a verification of remote file integrity. To rerun from the repository root in the current environment, use the project's Python environment and a short Python invocation importing `exp_c_fire_risk`; no new network access is needed while all cache files remain available. Preserve all original cache files before rerunning any downloader.

## Disease artifact checks

- Sum the `support` values for the 16 class rows in each per-class CSV: 3,563 for each model.
- Recompute macro-F1 as the unweighted mean of per-class F1 values: 0.9692186 for MobileNetV2 and 0.6534717 for the small CNN.
- Accuracy rows are 0.9761437 and 0.7575077 respectively.
- These checks validate summary arithmetic, not image-level predictions or split assignment.

The checked-in `exp_a_disease.py` declares defaults of image size 128, six epochs, batch size 32, seed 0, and dHash distance four. Do not report these as the actual historical run settings without the missing `results_a.json` manifest or equivalent execution record. The script writes `results_a.json`, but that file is not present with the current CSVs and images.

## Figure and manuscript build

Run `python build_smartagro_paper.py` from the repository root to create figures from supplied CSV/JSON/PNG artifacts, regenerate the HTML and audit sidecars, and build the editable OOXML document through `build_smartagro_docx.py`. The script does not train models or fetch data. Render the HTML source to PDF with `outputs/render_smartagro_paper.ps1`, which uses headless Chrome and print CSS. The final PDF has 24 pages total; the main manuscript through Conclusion is 22 pages when References and Appendix A are excluded. The editable DOCX is generated from the same HTML content and includes the same figures and tables; pagination may vary slightly in a different Word version. The final page counts are recorded in `quality_review.md`.

## Reproducibility omissions to address in a future experiment

Record source request URLs without secrets, response hashes, product versions, exact time standards, dataset and image hashes, package lock, Python/TensorFlow/scikit-learn versions, hardware, run command, random seeds, model checkpoints, split assignments, per-example scores, numeric confusion matrices, and full training logs. For fire inputs, explicitly set or normalize UTC before joining POWER dates to FIRMS acquisition dates. For future deployment evaluation, use weather forecasts available at the prediction issue time rather than retrospective daily observations.
"""
(OUT / "reproducibility.md").write_text(REPRO, encoding="utf-8")


LIMITS = """# Limitations and future work

## Current evidence limits

1. **Disease score provenance.** The supplied per-class reports support arithmetic checks and a test-support total of 3,563, but there is no run manifest, model checkpoint, image list, prediction table, split assignment, or numeric confusion matrix. The reported MobileNetV2 and small-CNN values therefore remain provisional.
2. **Subset scope and image domain.** Result files cover 16 classes; the local color tree has 38 classes and 54,305 files. The exact class filter is not logged. Controlled PlantVillage imagery does not establish field-photo performance. The local extracted repository has no explicit image-license file, and the canonical paper's count is 54,306, one higher than the local count.
3. **Fire target.** The label is a next-day, non-low-confidence VIIRS/S-NPP grid-cell detection. It is neither confirmed agricultural burning nor a field-level fire label. No cropland mask or independent cause label was used in the main run.
4. **Time and spatial structure.** The fire models are tested on 2022–2023 at grid cells already present in training. POWER daily fields default to local solar time; FIRMS dates are UTC; the script joins date strings without harmonization. The weekly bootstrap captures some temporal dependence but not spatial dependence among neighboring cells.
5. **Attribution and operations.** Fire models include recent detections, date, location, and weather; no ablation identifies a weather contribution. Inputs are retrospective, not issued forecasts. There is no lead-time study, field alert, operator response, latency, or cost analysis.
6. **No paired sensor or impact evidence.** No calibrated IoT readings, synchronized crop images, independent stress labels, farm actions, or outcomes are supplied. Fusion, prevention, loss reduction, and action verification remain proposed.

## Future work

1. Recover the exact disease invocation and source subset, then retain a complete manifest and per-image prediction output. If that cannot be recovered, conduct a new clearly versioned benchmark and label the old metrics historical.
2. Report full 38-class and explicitly defined subset results separately, with image/plant grouping, repeated seeds, uncertainty intervals, numeric confusion matrices, and a locked field-like test set.
3. Verify image redistribution rights and resolve the local 54,305 versus canonical 54,306 count before publishing data or claiming completeness.
4. Rebuild the fire dataset with file hashes, versioned inputs, independent hotspot/event labels where feasible, a cropland mask, UTC-aligned daily windows, spatial holdout, and forward temporal evaluation.
5. Run feature ablations for persistence, season, location, and weather; include forecast inputs available at alert time; evaluate calibration, lead time, alert burden, false alarms, and missed detections.
6. Collect calibrated sensor streams paired by timestamp and plot ID to plant images and expert-confirmed disease/stress assessments. Compare image-only, sensor-only, rule-based, and fused models on farm-separated external test data.
7. Evaluate the decision policy prospectively with farmer/extension participation, verified action logs, crop outcomes, connectivity, privacy, safety, costs, and an appropriate comparison with current practice.

The staged design and evidence gates are detailed in Table 10 of the manuscript. No additional experiments or results are inferred from these plans.
"""
(OUT / "limitations_and_future_work.md").write_text(LIMITS, encoding="utf-8")


STATUS = {
    "project": "SmartAgro research validation",
    "status": "component_evidence_audited_with_limits",
    "audit_date": "2026-10-22",
    "summary": "The earlier blocked status report predates later result files. Fire metrics reproduce exactly from supplied cache and code. Disease score artifacts remain provisional. Environmental stress, fusion, action verification, and field impact are not validated.",
    "evidence": {
        "plantvillage": {"local_color_classes": 38, "local_color_images": 54305, "canonical_count_in_source_paper": 54306, "license_file_found": False},
        "disease": {"status": "reported_but_unverified", "classes": 16, "test_support": 3563, "mobilenetv2_accuracy": 0.9761436991299467, "mobilenetv2_macro_f1": 0.9692185865844675, "small_cnn_accuracy": 0.7575077182149874, "small_cnn_macro_f1": 0.6534717249094737, "missing": ["results_a.json run manifest", "split/group IDs", "prediction-level outputs", "model checkpoints"]},
        "fire": {"status": "reproduced_from_supplied_cache", "mock": False, "cache_firms_csv": 91, "cache_power_json": 54, "test_positive_rate": 0.49544626593806923, "split_rows": {"train": 13176, "val": 3294, "test": 6588}, "replay_exact_match": True, "limitations": ["No remote cache hashes", "No cropland mask", "POWER local-solar dates joined to FIRMS UTC dates", "No spatial holdout", "Observed weather rather than forecast"]},
        "fire_debug": {"status": "not_interpretable", "reason": "Zero positive validation and test labels under fallback temporal split."},
        "environmental_stress": {"status": "not_validated", "reason": "No paired calibrated sensor data and confirmed stress labels."},
        "fusion": {"status": "not_run", "reason": "No synchronized image-sensor-label records."}
    }
}
(OUT / "experiment_status.json").write_text(json.dumps(STATUS, indent=2), encoding="utf-8")
(OUT / "experiment_status.md").write_text("""# SmartAgro experiment status — evidence audit update

Audit date: 22 October 2026.

The earlier status note said that no real metrics were valid because image loading and FIRMS access were blocked. That note predates the current disease output files and main fire result/cache. Its blanket status is stale.

## Current status

- **Disease classification:** per-class CSVs report MobileNetV2 accuracy 97.6144% / macro-F1 96.9219% and small-CNN accuracy 75.7508% / macro-F1 65.3472% on 16 classes with 3,563 summed test supports. Arithmetic is internally consistent, but `results_a.json`, predictions, split IDs, and model checkpoints are absent. Keep these values labelled provisional.
- **Satellite fire-risk assessment:** the main result uses `mock: false`. Replaying the 91 FIRMS CSV pulls and 54 POWER JSON grid files through `exp_c_fire_risk.py` reproduces split counts, prevalence, average precision, threshold metrics, and Brier scores exactly. The target is next-day non-low-confidence VIIRS detection, not confirmed agricultural burning. Cache provenance and UTC/local-solar date alignment remain unresolved.
- **Debug fire result:** not interpretable; validation and test have zero positives.
- **Environmental stress:** no paired calibrated sensor data or verified stress labels; not validated.
- **Multimodal fusion and actions:** no synchronized records, alert actions, or outcomes; not evaluated.

The local extracted PlantVillage color tree contains 38 classes and 54,305 images, one fewer than the 54,306 count in the original paper. No explicit data-license file was found in the extracted repository. The local tree is separate from the previously observed text-only Hugging Face loader response.

See `evidence_audit.md` for file-by-file evidence, numerical checks, and unresolved claims. See `reproducibility.md` for the cache replay and recorded software versions.
""", encoding="utf-8")


RENDER_PS = r'''$ErrorActionPreference = "Stop"
$htmlPath = Join-Path $PSScriptRoot "SmartAgro_Research_Paper.html"
$pdfPath = Join-Path $PSScriptRoot "SmartAgro_Research_Paper.pdf"
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$profilePath = Join-Path $env:TEMP "smartagro_pdf_profile"
$fileUrl = "file:///" + ($htmlPath -replace "\\", "/")
& $chromePath --headless=new --disable-gpu --disable-gpu-compositing --disable-gpu-rasterization --disable-extensions --no-first-run --no-default-browser-check --user-data-dir=$profilePath --print-to-pdf=$pdfPath $fileUrl
if ($LASTEXITCODE -ne 0) { throw "Chrome PDF rendering failed with exit code $LASTEXITCODE" }
Write-Output "PDF=$pdfPath"
'''
(OUT / "render_smartagro_paper.ps1").write_text(RENDER_PS, encoding="utf-8")

print(f"Wrote {html_path}")
print(f"Created {len(list(FIG.glob('fig*.png')))} PNG figures and matching PDFs")
print(f"Disease support sum: {int(mobile.loc[class_names, 'support'].sum())}")
print(f"Fire prevalence: {setup['base_rate_test']:.10f}")
from build_smartagro_docx import main as build_docx
build_docx()
