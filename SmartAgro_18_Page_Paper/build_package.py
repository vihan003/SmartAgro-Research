from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from lxml import html as lxml_html
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA = OUT / "SmartAgro_Figure_Source_Data"
FIG = OUT / "SmartAgro_Figures"
CONTENT = OUT / "manuscript_content.html"
HTML_OUT = OUT / "SmartAgro_Full_Research_Paper.html"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_source(src: str, dest: str | None = None) -> Path:
    source = ROOT / src
    target = DATA / (dest or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def prepare_figure_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source_inputs = [
        "outputs/results_a/per_class_grouped__mobilenetv2.csv",
        "outputs/results_a/per_class_grouped__small_cnn.csv",
        "data/environmental/processed/smartagro_iot.csv",
        "paper_results/environmental_monitoring/dataset_summary.csv",
        "paper_results/environmental_monitoring/sensor_statistics.csv",
        "paper_results/environmental_monitoring/evaluation_summary.csv",
        "paper_results/environmental_monitoring/stress_category_summary.csv",
        "paper_results/environmental_monitoring/temporal_cv_summary.csv",
        "paper_results/fire_risk/model_comparison.csv",
        "paper_results/fire_risk/evaluation_details.csv",
        "paper_results/fire_risk/dataset_summary.csv",
    ]
    for source in source_inputs:
        copy_source(source)

    # Primary fire models retain the original validation-selected operating protocol.
    results = json.loads((ROOT / "outputs/results_c/results_c.json").read_text(encoding="utf-8"))
    primary_rows = []
    for name, model in results["models"].items():
        ci = model.get("AUPRC_CI95_block_bootstrap", [None, None])
        primary_rows.append({
            "model": name, "average_precision": model["AUPRC"],
            "ci95_low": ci[0], "ci95_high": ci[1],
            "precision": model.get("precision"), "recall": model.get("recall"),
            "missed_event_rate": model.get("missed_event_rate"),
            "false_alarms_per_100_negative_cell_days": model.get("false_alarms_per_100_negative_cell_days"),
            "alert_share_pct": model.get("alert_share_pct"), "brier": model.get("brier"),
            "protocol": "Primary; threshold selected on 2021 validation to target 80% cumulative recall",
        })
    pd.DataFrame(primary_rows).to_csv(DATA / "fire_primary_results.csv", index=False)
    aux = pd.read_csv(ROOT / "paper_results/fire_risk/model_comparison.csv")
    aux_names = {
        "logistic regression (0.5 threshold)": "Logistic regression",
        "histogram gradient boosting (0.5 threshold)": "Histogram gradient boosting",
        "random forest (0.5 threshold)": "Random forest",
        "xgboost (0.5 threshold)": "XGBoost",
    }
    aux = aux[aux.model.isin(aux_names)].copy()
    aux["model_display"] = aux.model.map(aux_names)
    aux.to_csv(DATA / "fire_auxiliary_results.csv", index=False)

    # Reconstruct source-backed regional summaries and the primary-model PR curve offline.
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve
    sys.path.insert(0, str(ROOT))
    import exp_c_fire_risk as fire

    cache = ROOT / "outputs/results_c/cache"
    firms = []
    for p in sorted(cache.glob("firms_*.csv")):
        d = pd.read_csv(p)
        if "confidence" in d:
            d = d[d.confidence.astype(str).str.lower() != "l"]
        firms.append(d)
    fires = pd.concat(firms, ignore_index=True).drop_duplicates()
    monthly = fires.copy()
    monthly["acq_date"] = pd.to_datetime(monthly.acq_date)
    monthly = monthly[monthly.acq_date.dt.month.isin([10, 11])]
    monthly = monthly.groupby([monthly.acq_date.dt.year.rename("year"), monthly.acq_date.dt.month.rename("month")]).size().rename("hotspot_records").reset_index()
    monthly.to_csv(DATA / "fire_oct_nov_counts.csv", index=False)

    power = []
    for p in sorted(cache.glob("power_*.json")):
        js = json.loads(p.read_text(encoding="utf-8"))
        d = pd.DataFrame(js["properties"]["parameter"])
        d.index = pd.to_datetime(d.index, format="%Y%m%d")
        d = d.replace(-999, np.nan).reset_index().rename(columns={"index": "date"})
        lat, lon = map(float, p.stem.replace("power_", "").split("_"))
        d["lat"], d["lon"] = lat, lon
        d["cell"] = fire.cell_id(np.array([lat]), np.array([lon]))[0]
        power.append(d)
    wx = pd.concat(power, ignore_index=True)
    model_df = fire.build(fires, wx)
    cell_rates = model_df.groupby("cell").agg(
        lat=("lat", "first"), lon=("lon", "first"),
        positive_days=("y", "sum"), cell_days=("y", "size"),
        next_day_detection_rate=("y", "mean"),
    ).reset_index()
    cell_rates.to_csv(DATA / "fire_grid_cell_rates.csv", index=False)

    tr = model_df[model_df.year <= 2020]
    te = model_df[model_df.year > 2021].reset_index(drop=True)
    gbt = HistGradientBoostingClassifier(max_depth=3, learning_rate=.08, max_iter=150, random_state=0)
    gbt.fit(tr[fire.FEATS], tr.y)
    prob = gbt.predict_proba(te[fire.FEATS])[:, 1]
    ap = average_precision_score(te.y, prob)
    stored = results["models"]["gradient-boosted trees"]["AUPRC"]
    if abs(ap - stored) > 1e-10:
        raise RuntimeError(f"Figure replay AP mismatch: {ap} != {stored}")
    precision, recall, _ = precision_recall_curve(te.y, prob)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(DATA / "fire_gbt_pr_curve.csv", index=False)

    manifest = {
        "purpose": "Source data and file hashes for figures in the manuscript",
        "project_root": "Parent project folder; no API keys are used or included",
        "source_files": [{"path": s, "sha256": sha256(ROOT / s)} for s in source_inputs],
        "derived_files": [
            "fire_primary_results.csv: results_c.json primary models and operating metrics",
            "fire_auxiliary_results.csv: fixed 0.5-threshold comparison table",
            "fire_oct_nov_counts.csv: monthly counts from cached FIRMS detections, October-November only",
            "fire_grid_cell_rates.csv: next-day label rates reconstructed from FIRMS/POWER cache",
            "fire_gbt_pr_curve.csv: deterministic seed-0 replay; AP checked against results_c.json",
        ],
        "figure_builder": "recreate_figures.py; uses only CSVs in this directory",
    }
    (DATA / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA / "README.md").write_text(
        "# Figure source data\n\n"
        "CSV files here are either direct copies of checked-in project result tables or small derived summaries from the local FIRMS/POWER cache. "
        "The complete raw datasets are not repackaged. `recreate_figures.py` regenerates all PNG and PDF figure exports using only these files. "
        "Source paths, SHA-256 digests, and derived-file descriptions are in `source_manifest.json`. The motor-monitor source has a non-commercial license; "
        "only aggregate evaluation summaries are included. Disease image files are not redistributed.\n",
        encoding="utf-8",
    )


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
    separate = OxmlElement("w:fldChar"); separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t"); text.text = "1"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for e in (begin, instr, separate, text, end): run._r.append(e)


def build_docx() -> None:
    tree = lxml_html.parse(str(CONTENT))
    body = tree.find(".//body")
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.72)
    sec.left_margin = Inches(.78); sec.right_margin = Inches(.78)
    sec.header_distance = Inches(.35); sec.footer_distance = Inches(.35)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"; normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(5)
    for style_name, size in (("Title", 17), ("Heading 1", 13), ("Heading 2", 11), ("Heading 3", 10)):
        st = doc.styles[style_name]
        st.font.name = "Times New Roman"; st.font.size = Pt(size)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)

    def add_table(node):
        trs = node.xpath(".//tr")
        if not trs: return
        headers = trs[0].xpath("./th|./td")
        ncols = len(headers)
        t = doc.add_table(rows=0, cols=ncols)
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for ri, tr in enumerate(trs):
            cells = tr.xpath("./th|./td")
            row = t.add_row()
            for ci in range(ncols):
                cell = row.cells[ci]
                cell.text = cells[ci].text_content().strip() if ci < len(cells) else ""
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    for r in p.runs:
                        r.font.name = "Times New Roman"; r.font.size = Pt(8)
                        if ri == 0: r.bold = True
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def add_inline(paragraph, node, bold=False, italic=False):
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        bold = bold or tag in ("b", "strong")
        italic = italic or tag in ("i", "em")
        if tag == "br":
            paragraph.add_run().add_break()
            return
        if node.text:
            run = paragraph.add_run(node.text)
            run.bold = bold; run.italic = italic
        for child in node:
            add_inline(paragraph, child, bold, italic)
            if child.tail:
                run = paragraph.add_run(child.tail)
                run.bold = bold; run.italic = italic

    for node in body:
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag == "h1":
            p = doc.add_paragraph(style="Title"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(node.text_content().strip())
        elif tag == "h2":
            doc.add_heading(node.text_content().strip(), level=1)
        elif tag == "h3":
            doc.add_heading(node.text_content().strip(), level=2)
        elif tag == "p":
            p = doc.add_paragraph()
            add_inline(p, node)
            p.paragraph_format.space_after = Pt(5)
            if "author" in (node.get("class") or "") or "affiliation" in (node.get("class") or ""):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif tag == "table":
            add_table(node)
        elif tag == "figure":
            img = node.find("img")
            if img is not None:
                img_path = OUT / img.get("src")
                if img_path.exists():
                    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.keep_with_next = True
                    p.add_run().add_picture(str(img_path), width=Inches(6.35))
            cap = node.find("figcaption")
            if cap is not None:
                p = doc.add_paragraph(cap.text_content().strip())
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = False
                for r in p.runs: r.font.size = Pt(8)
        elif tag in ("ul", "ol"):
            for li in node.findall("li"):
                p = doc.add_paragraph(li.text_content().strip(), style="List Bullet" if tag == "ul" else "List Number")
        elif tag == "div" and node.get("class") == "table-caption":
            p = doc.add_paragraph(node.text_content().strip())
            p.paragraph_format.keep_with_next = True
            for r in p.runs: r.bold = True; r.font.size = Pt(8.5)

    doc.save(OUT / "SmartAgro_Full_Research_Paper.docx")


def build_html() -> None:
    tree = lxml_html.parse(str(CONTENT))
    body = tree.find(".//body")
    inner = "\n".join(lxml_html.tostring(child, encoding="unicode", method="html") for child in body)
    css = """
    @page { size: A4; margin: 16mm 18mm 17mm 18mm; }
    * { box-sizing: border-box; }
    body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.3pt; line-height: 1.28; color: #20282e; }
    h1 { font-size: 19pt; line-height: 1.15; text-align: center; margin: 0 0 14pt; color: #18324b; }
    h2 { font-size: 14pt; color: #18324b; margin: 17pt 0 7pt; page-break-after: avoid; }
    h3 { font-size: 11pt; color: #285b78; margin: 12pt 0 5pt; page-break-after: avoid; }
    p { margin: 0 0 7pt; text-align: justify; orphans: 3; widows: 3; }
    p.author, p.affiliation { text-align: center; font-size: 9.5pt; margin-bottom: 3pt; }
    figure { margin: 11pt auto 12pt; text-align: center; page-break-inside: avoid; }
    figure img { display: block; max-width: 100%; max-height: 220mm; margin: 0 auto 5pt; object-fit: contain; }
    figcaption { font-family: Georgia, 'Times New Roman', serif; font-size: 8.5pt; line-height: 1.2; text-align: left; }
    .table-caption { font-size: 9pt; font-weight: bold; margin: 10pt 0 3pt; page-break-after: avoid; }
    table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.15; margin: 0 0 10pt; page-break-inside: avoid; }
    th, td { border: .5pt solid #aab5bc; padding: 4pt; vertical-align: top; }
    th { background: #e8eff3; text-align: left; color: #18324b; }
    @media print { a { color: inherit; text-decoration: none; } }
    """
    rendered = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>SmartAgro Full Research Paper</title><style>" + css + "</style></head><body>" + inner + "</body></html>"
    HTML_OUT.write_text(rendered, encoding="utf-8")


def main() -> None:
    prepare_figure_data()
    subprocess.run([sys.executable, str(DATA / "recreate_figures.py")], check=True, cwd=OUT)
    build_docx()
    build_html()
    print(json.dumps({"html_source": str(HTML_OUT), "docx": str(OUT / "SmartAgro_Full_Research_Paper.docx"),
                      "figures": len(list(FIG.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
