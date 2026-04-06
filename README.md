# SEDS 537 Machine Learning Take-Home Midterm

This repository contains the code, experiments, and report materials for the SEDS 537 take-home midterm. The assignment is individual and requires a reproducible workflow, a structured LaTeX report, and a repository link as part of the submission.

The project covers five core machine learning tasks:
- Regression on the California Housing dataset
- Imbalanced classification with resampling techniques
- Dimensionality reduction and visualization on MNIST
- Unsupervised clustering analysis
- Neural network comparison on Fashion-MNIST

The final submission should include reproducible notebooks or Python scripts, figures and tables supporting the analysis, and a report discussing methodology, evaluation, and findings. According to the assignment brief, the deadline is **April 27, 2026 at 23:59**, and the submission archive should be named `SEDS537 Midterm <StudentID>.zip`.

Suggested script entry points:

```bash
python -m src.q1_regression.run
python -m src.q2_classification.run
python -m src.q3_dimensionality_reduction.run
python -m src.q4_clustering.run
python -m src.q5_neural_networks.run
```

Project layout:
- `src/` contains the implementation for each question and shared utilities in `src/common/`
- `data/` stores raw, interim, and processed datasets
- `results/` stores generated figures and tables
- `docs/report/` stores the LaTeX report
- `models/saved/` stores trained model artifacts when needed

Assignment reference: [`docs/SEDS567_takeHome.pdf`](docs/SEDS567_takeHome.pdf)
