# SmartAgro research paper package

The main deliverables are the editable Word manuscript and the 18-page PDF:

- `SmartAgro_Full_Research_Paper.docx`
- `SmartAgro_Full_Research_Paper.pdf`

The package also includes the styled HTML source, manuscript source HTML, 12 PNG/PDF figures and their generation script, copied/derived figure CSVs with a hash manifest, a references verification note, a research quality audit, and a reproducibility/claims checklist.

## Rebuild

From the project root, run:

```powershell
.venv\Scripts\python.exe SmartAgro_18_Page_Paper\build_package.py
```

The script rebuilds figure source data, all figures, the DOCX, and the styled HTML. To recreate the PDF, print the HTML with Chrome headless on A4 with browser headers and footers disabled. The delivered PDF contains 18 pages.

## Submission edits

Replace the author, affiliation, and contact placeholders. Confirm all project-specific author contributions and institutional declarations. The manuscript intentionally labels disease results as reported/unverified at run level, environmental models as exploratory, and the fire target as next-day satellite detection rather than confirmed agricultural fire.
