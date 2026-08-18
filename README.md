# Multi-Omics Clinical Integration Pipeline

## Overview

This repository contains a multi-omics and clinical integration pipeline designed to process genomic, transcriptomic, and clinical data in a modular workflow. The project includes data collection, preprocessing, integration, export, and validation components.

## Repository Structure

- `src/` — core pipeline source code
- `config/` — YAML configuration files for pipeline execution
- `data/` — raw, processed, and external datasets
- `demo_simple.py` — simplified demo pipeline example
- `README.md` — project documentation
- `requirements.txt` — Python dependencies
- `tests/` — testing utilities and coverage scaffolding
- `results/` — pipeline output data
- `logs/` — execution logs and debugging information

## Installation

1. Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Pipeline behavior is controlled by YAML files in `config/`, including:

- `config/config.yaml` — main pipeline configuration
- `config/pipeline_config.yaml` — integration and export settings

Update config values before running the pipeline to match your dataset paths, export options, and analysis goals.

## Usage

Run the main pipeline from the repository root:

```bash
python src/pipeline.py --omic-data demo_expression_data.csv --clinical-data demo_clinical_data.csv --output results
```

Alternatively, use the demo script for a simplified end-to-end example:

```bash
python demo_simple.py
```

## Demo Data

The repository includes demonstration files:

- `demo_expression_data.csv`
- `demo_clinical_data.csv`

These files are intended for quick validation and functional tests of the pipeline.

## Output

Pipeline output is stored under `results/` and may include:

- standardized CSV export
- JSON export
- additional `test_results_*` directories for validation runs

Do not commit generated output files to the repository unless they are part of a curated example dataset.

## Large Data Management

This repository contains data and results that may be very large. To avoid GitHub errors related to file size limits and push timeouts, follow these best practices:

- do not commit raw data files (`data/raw/...`) to the main repository
- ignore generated or output folders:
  - `data/`
  - `results/`
  - `logs/`
- use Git LFS for large files that need tracking

## Git LFS recommended

If you need to keep large files in history, use Git LFS:

```bash
git lfs install
git lfs track "*.txt"
git lfs track "*.csv"
git lfs track "*.seg"
git lfs track "*.png"
```

Then add and commit:

```bash
git add .gitattributes
git add <your-large-files>
git commit -m "Track large files with Git LFS"
```

## Quick recommendation

- Keep source code in `src/`
- Keep large data outside the repository or from external sources
- Do not version large files that exceed 100 MB
- Use `git status` before each push to verify the state

## Testing

Run available tests using Python or your preferred test runner:

```bash
python test_final.py
python test_pipeline_complete.py
```
