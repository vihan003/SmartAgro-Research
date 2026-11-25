# Research quality and PDF layout audit

**Artifact:** *SmartAgro: An Evidence-Audited Framework Integrating Crop-Disease Classification, Environmental Monitoring, and Regional Satellite Hotspot Risk Assessment*  
**Audit date:** 20 November 2026  
**Current PDF:** 18 pages  
**Scope:** factual consistency against checked-in project outputs, provenance and claims, figure sources, reference list, and rendered-page layout.

## Audit summary

The manuscript reports three evidence levels separately. The regional fire benchmark is described as reproduced because the primary histogram-gradient-boosting result was replayed from the cached FIRMS/POWER inputs and its average precision matched the stored result. Disease metrics are reported from project CSV summaries and explicitly marked as not independently replayed. Environmental sensor summaries and pump-status results are presented as descriptive or exploratory because the source labels, test support, timestamps, and measurement units do not support strong field claims.

The paper contains 12 figures, 7 tables, 15 references, an abstract, research questions, methods, results, discussion, limitations, validation roadmap, conclusion, and data availability/responsible-use statements. The PDF was rendered with Chrome headless print output. All 18 pages were reviewed in a page contact sheet; the title page and final references page were also inspected at page scale. Page headers and browser print footers are disabled. The final references continue onto page 18 without an isolated final line.

## Evidence and claim checks

- Disease tables cover 16 classes and 3,563 summed test examples per model. The stated accuracy and macro-F1 values match the project result summaries. The paper identifies missing checkpoints, prediction files, exact split IDs, and run manifest.
- Disease script settings are described as checked-in defaults, not proven exact settings for the CSV run. The supplied confusion-matrix images are not used to infer numeric counts.
- Environmental dataset row counts and source caveats match local documentation. The Kaur et al. moisture values are described on the raw sensor scale rather than as percentages. Synthetic playback timestamps are identified.
- The predictive-irrigation test support is reported as 142 off and 2 on. Its perfect metrics are not presented as validated generalization.
- The SmartAgroTech sensor archive is treated as 101 timestamp-matched readings without stress labels. The 97 Low and 4 High results are identified as fixed-rule outputs, not diagnosed crop stress.
- Fire target, region, season, cell count, temporal split, feature set, metrics, bootstrap intervals, and operating-point results are aligned with the local result JSON and source cache. The model target is consistently described as a next-day VIIRS detection.
- The paper notes observed weather, local-solar-time/UTC alignment, lack of spatial holdout, neighboring-cell dependence, absence of a crop mask, and overlap of logistic-regression and gradient-boosting uncertainty intervals.
- The auxiliary random-forest and XGBoost values are labeled descriptive, without claiming statistical superiority.
- Twelve figures are generated from the included CSVs and figure script. The source manifest records SHA-256 hashes for copied source result files and identifies derived summaries.

## Layout and production checks

- PDF page count: **18**.
- Extractable PDF text: approximately **6,700 words** (figure text and labels are not fully counted by PDF text extraction).
- Figures: **12 PNG** exports and matching **12 PDF** exports.
- Tables: **7** in the manuscript.
- References: **15**.
- The PDF uses A4 layout, readable serif body text, figure captions, and no browser timestamp/URL footer.
- DOCX is editable and contains the manuscript text, tables, figures, and italic formatting for titles and journal/book elements.

## Items to resolve before external submission

1. Replace author, affiliation, and correspondence placeholders with verified details.
2. Confirm whether the 16-class disease CSV corresponds to the checked-in experiment script and supply the precise class list, split manifest, run configuration, model checkpoints, and per-example predictions if available.
3. Recheck every reference and dataset attribution against the exact records required by the target journal, then apply that journal's citation and manuscript template.
4. Confirm permissions and license obligations for any images or source data added to a submission. The Kaur et al. dataset is marked CC BY-NC 3.0; raw files are not bundled here.
5. If describing the system as deployed or validated in a farm setting, add new primary evidence. This manuscript does not establish that claim.

## Build files

- `build_package.py` prepares copied and derived figure data, runs `SmartAgro_Figure_Source_Data/recreate_figures.py`, and creates DOCX and styled HTML.
- The PDF is printed from `SmartAgro_Full_Research_Paper.html` with Chrome headless.
- `SmartAgro_Figure_Source_Data/source_manifest.json` records source paths and hashes.
