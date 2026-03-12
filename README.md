# Classifiers with Dimensionality Reduction Comparisons

A comparative study of classification performance using different dimensionality reduction techniques on high-dimensional datasets.

## Overview

This project evaluates how different dimensionality reduction methods affect classification accuracy across multiple datasets and classifiers. It generates metrics (accuracy, precision, recall) at various reduced dimensions and outputs results as JSON files.

## Datasets

| Dataset | Features | Description |
|---------|----------|-------------|
| Olivetti Faces | 4,096 | 400 grayscale face images (64x64), 40 subjects |
| MNIST | 784 | Handwritten digit images (28x28) |
| Fashion-MNIST | 784 | Clothing item images (28x28) |
| 20 Newsgroups | 5,000 | Text documents (TF-IDF features) |

## Dimensionality Reduction Methods

- **None** - Baseline with no reduction
- **PCA** - Principal Component Analysis
- **Diffusion Maps** - Nonlinear manifold learning
- **MDS** - Multidimensional Scaling

## Classifiers

- **KNN** - K-Nearest Neighbors (k=3)
- **Logistic Regression** - Multinomial logistic regression
- **Naive Bayes** - Gaussian Naive Bayes

## Project Structure

```
classifier_dr_comparison/
├── data.py              # Dataset loaders
├── dim_red.py           # Dimensionality reduction methods
├── classifier.py        # Classification algorithms
├── experiment.py        # Main experiment pipeline
├── generate_latex_table.py  # LaTeX table generation
├── results/             # JSON output files
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Output

Results are saved as JSON files in `results/` with the naming convention:
```
{dataset}_{reduction_method}_{classifier}.json
```

Each file contains per-dimension metrics and summary statistics including best accuracy and optimal dimension count.