# Reproducibility and claims checklist

Use this checklist when discussing or revising the manuscript. It separates findings supported by the current files from claims that require new evidence.

## Claims supported by the current project records

- SmartAgro is a proposed modular framework containing crop-disease image classification, environmental monitoring, and regional satellite hotspot analysis.
- The three modules were evaluated independently. No synchronized end-to-end deployment or integrated field trial is demonstrated.
- The disease result files report 97.61% accuracy and 96.92% macro-F1 for MobileNetV2, and 75.75% accuracy and 65.35% macro-F1 for a small CNN, across 16 labels and 3,563 summed test examples. Describe these as reported grouped-split results; the exact run cannot currently be independently reproduced.
- The public environmental sources contain 960, 3,000, and 101 matched rows, respectively. Preserve each source's timestamp and measurement limitations.
- The predictive-irrigation held-out set contains only two pump-on examples. Its perfect metrics are not sufficient evidence of reliable generalization.
- On the 101-reading archive, the preset threshold function produces 97 Low and 4 High categories. These are rule outputs; the archive has no independent crop-stress labels.
- The primary fire model predicts the next-day presence of at least one non-low-confidence VIIRS/S-NPP detection in a regional grid cell. It does not directly predict confirmed agricultural fires.
- The replayed primary test AP for histogram gradient boosting is 0.872 (95% weekly block-bootstrap interval 0.801–0.913); logistic regression is 0.853 (0.797–0.894), and persistence is 0.741 (0.690–0.782). The learned-model intervals overlap.
- Fire features use daily observed POWER weather, not forecast weather. The temporal split does not hold out cells spatially.

## Claims that need additional evidence

- Disease diagnosis accuracy on new farms, field photographs, cameras, seasons, or expert-labeled cases.
- Irrigation savings, improved crop yield, reduced stress, or improved pump reliability.
- A validated crop-stress diagnosis from the fixed rule categories.
- Confirmed fire occurrence, fire cause, crop-residue burning, crop-field risk, or operational fire warnings.
- Prospective forecasting performance using weather forecasts and correctly aligned timestamps.
- Generalization to new locations, crops, devices, or satellite grid cells.
- End-to-end integration, real-time IoT operation, user benefit, safety, or economic impact.

## Reproduction steps

1. From the project root, run `\.venv\Scripts\python.exe SmartAgro_18_Page_Paper\build_package.py`.
2. Inspect `SmartAgro_Figure_Source_Data/source_manifest.json` and ensure the listed input files and hashes match the intended project revision.
3. To regenerate figures alone, run `\.venv\Scripts\python.exe SmartAgro_18_Page_Paper\SmartAgro_Figure_Source_Data\recreate_figures.py`.
4. Print `SmartAgro_Full_Research_Paper.html` to PDF on A4 with browser headers and footers disabled; the delivered PDF has 18 pages.
5. For a true disease rerun, first recover the exact class manifest, image-to-group assignments, run configuration, environment versions, trained weights, and per-image predictions. Script defaults alone do not establish the settings of the stored summaries.
6. For field or operational claims, collect prospective, correctly timestamped, locally calibrated data and use an evaluation split that matches the intended deployment location and time horizon.

## File and licensing notes

- Figure CSVs and derived summaries are provided in `SmartAgro_Figure_Source_Data/`.
- Raw disease images and raw motor-monitor records are not redistributed.
- Cite the Mendeley dataset records and check their license terms before reusing or sharing source files. The Kaur et al. dataset is marked CC BY-NC 3.0.
- Do not include credentials or private API tokens in a public archive.
