# Overleaf Package: Scientific Model Demonstration

## Main file
- `SCIENTIFIC_MODEL_DEMONSTRATION.tex`

## Contents
- `outputs/paper_figures/` -> all PNG figures used by the manuscript
- `data/experimental/100425_chip1_gap3.tsv` -> specimen dataset used in the derivation/fitting discussion
- `presets/resistance_100425_chip1_gap3.json` -> fitted parameter preset referenced in the manuscript
- `scripts/generate_paper_figures.py` -> script that regenerates the manuscript figures from dataset/preset
- `references/collective_Dynamics.pdf`, `references/2582_1.pdf` -> source papers
- `SCIENTIFIC_MODEL_DOCUMENTATION.md` -> companion prose documentation

## Overleaf upload
1. Zip this entire folder.
2. Upload the zip to Overleaf.
3. Set `SCIENTIFIC_MODEL_DEMONSTRATION.tex` as the main file.

## Notes
- The current manuscript includes bibliography entries directly in the `.tex` file (no separate `.bib` needed).
- Image paths are already relative to this package structure.
