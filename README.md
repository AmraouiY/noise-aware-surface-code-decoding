# Noise-Aware Surface Code Decoding

This repository contains code and notebooks for studying noise-aware decoding strategies for the surface code and repetition code.

## Repository structure

- `notebooks/`
  - `01_repetition_code.ipynb` — analysis and experiments for the repetition code
  - `02_surface_code_baseline.ipynb` — baseline surface code simulation and decoding
- `src/`
  - `repetition.py` — repetition code simulation utilities
  - `surface_baseline.py` — surface code baseline simulation utilities
  - `plotting.py` — plotting and visualization helpers
- `data/` — raw input / generated datasets
- `figures/` — output figures for the report
- `report/` — report drafts and writeups
- `requirements.txt` — Python dependencies

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- Add simulation code to `src/` and use `notebooks/` for interactive analysis.
- Keep datasets in `data/` and generated visuals in `figures/`.
