# SEDS 537 Machine Learning Take-Home Midterm

This repository contains the code, generated outputs, and LaTeX report for the SEDS 537 take-home midterm project. The work is organized question by question so that each experiment can be rerun from script entry points under `src/`.

The assignment covers five topics:
- Question 1: Regression on California Housing
- Question 2: Imbalanced classification on credit card fraud data
- Question 3: Dimensionality reduction and visualization on MNIST
- Question 4: Clustering on Mall Customer Segmentation data
- Question 5: Neural networks on Fashion-MNIST

Assignment reference: [`docs/SEDS567_takeHome.pdf`](docs/SEDS567_takeHome.pdf)

## Quick Start

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.11` is not installed on macOS:

```bash
brew install python@3.11
```

On macOS, `xgboost` may require OpenMP:

```bash
brew install libomp
```

After setup, confirm the interpreter:

```bash
python --version
```

Expected version: `Python 3.11.x`

## Important Notes

- Run project scripts with `python -m ...`, not `python path/to/file.py`
- Use the virtual environment interpreter after activation
- The project uses `RANDOM_STATE = 42` from `src/common/config.py` where applicable
- Keep preprocessing and dataset splits consistent across models unless the comparison explicitly changes one factor
- The test set should be used only for final evaluation

## Dataset Setup

Place manually downloaded datasets in `data/raw/` using these filenames:

- Question 1: `data/raw/california_housing.csv`
- Question 2: `data/raw/creditcard.csv`
- Question 4: `data/raw/Mall_Customers.csv`

Datasets handled by code:

- Question 3: MNIST is downloaded automatically by the script
- Question 5: Fashion-MNIST is downloaded automatically by the script

## How To Run

Run each question from the repository root:

```bash
python -m src.q1_regression.run
python -m src.q2_classification.run
python -m src.q3_dimensionality_reduction.run
python -m src.q4_clustering.run
python -m src.q5_neural_networks.run
```

## What Each Script Produces

- `src.q1_regression.run`
  Generates EDA outputs, regression metric tables, and residual plots under `results/figures/q1/` and `results/tables/q1/`
- `src.q2_classification.run`
  Generates baseline, SMOTE, and undersampling results, confusion matrices, and ROC outputs under `results/figures/q2/` and `results/tables/q2/`
- `src.q3_dimensionality_reduction.run`
  Generates PCA and t-SNE plots plus k-NN comparison tables under `results/figures/q3/` and `results/tables/q3/`
- `src.q4_clustering.run`
  Generates K-Means, Agglomerative, and DBSCAN plots, summaries, and evaluation tables under `results/figures/q4/` and `results/tables/q4/`
- `src.q5_neural_networks.run`
  Generates Fashion-MNIST sample images, training curves, test metrics, confusion matrices, and misclassified-example figures under `results/figures/q5/` and `results/tables/q5/`

## Report Build

The LaTeX report is stored in `docs/report/`.

Compile it from the report directory:

```bash
cd docs/report
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The second run resolves figure and table references.

## Current Status

- Question 1: implemented
- Question 2: implemented
- Question 3: implemented
- Question 4: implemented
- Question 5: implemented

## Project Layout

- `src/` contains question-specific code and shared helpers under `src/common/`
- `data/` stores raw, interim, and processed data
- `results/` stores generated figures and tables
- `docs/report/` contains the LaTeX report and compiled PDF
- `models/saved/` is reserved for saved model artifacts

## Submission Reminder

According to the assignment brief, the submission should include the repository link, the report, and supporting code and outputs. The archive name should follow the assignment instructions exactly.
