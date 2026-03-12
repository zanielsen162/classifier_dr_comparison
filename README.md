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
├── results/             # JSON output files
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Experiment

Run a single experiment with specific parameters:

```bash
python experiment.py --data faces --reducer pca --classifier knn --dims 2 5 10 20 50
```

**Arguments:**

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--data` | `faces`, `newsgroups`, `fashion-minst`, `minst` | `faces` | Dataset to use |
| `--reducer` | `pca`, `diff_map`, `none`, `mds` | `pca` | Dimensionality reduction method |
| `--classifier` | `knn`, `bayes`, `regression` | `knn` | Classification algorithm |
| `--dims` | integers | `2 5 10 20 50` | Dimensions to test |
| `--output` | path | `results` | Output directory |

### Full Experiment Suite

Run all combinations of datasets, reducers, and classifiers:

```bash
python experiment.py --full True
```

Optionally limit to specific datasets or reducers:

```bash
python experiment.py --full True --datasets faces minst --reducer pca diff_map
```

## Output

Results are saved as JSON files in `results/` with the naming convention:
```
{dataset}_{reduction_method}_{classifier}.json
```

Each file contains per-dimension metrics and summary statistics including best accuracy and optimal dimension count.
