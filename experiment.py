from data import load_olivetti, load_newsgroups, load_minst, load_fashion_minst, Data
from dim_red import run_pca, run_mds, run_diffusion_map, run_isomap
from classifier import KNN, logreg, naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import json
import os

LOADERS = {
    'faces': load_olivetti,
    'newsgroups': load_newsgroups,
    'fashion-minst': load_fashion_minst,
    'minst': load_minst,
}

REDUCERS = {
    'isomap': run_isomap,
    'mds': run_mds,
    'diff_map': run_diffusion_map,
    'pca': run_pca,
}

CLASSIFIERS = {
    'knn': KNN,
    'bayes': naive_bayes,
    'regression': logreg
}

def get_classifier_results(
        data: Data, 
        dim_red: str, 
        dimensions: List[int], 
        classifier_type: str
) -> Dict:
    """
    Takes data_name, dim_red, dimensions, and classifier type
    Loads data, performs reduction methods, runs classifier on different dimensions
    Returns result of each dimension (predictions on test set)
    """
    reducer = REDUCERS.get(dim_red, run_pca)
    classifier = CLASSIFIERS.get(classifier_type, logreg)

    reduced = reducer(data)

    return {
        d: classifier(
            reduced.train_components[:, :d],
            data.y_train,
            reduced.test_components[:, :d]
        ) for d in dimensions
    }

def analyze_data(
        results: Dict[int, np.ndarray], 
        y_true: np.ndarray
) -> Dict:
    """
    Analyze classifier results across different dimensions.
    Takes results as input and actual truth labels
    Returns dict with metrics for each dimension and summary stats
    """

    # basic metrics by dimension
    metrics = {}
    for d, y_pred in results.items():
        metrics[d] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
    
    # summary of stats for all dimensions
    dimensions = sorted(metrics.keys())
    accuracies = [metrics[d]['accuracy'] for d in dimensions]
    f1_scores = [metrics[d]['f1'] for d in dimensions]
    
    summary = {
        'best_accuracy_dim': dimensions[np.argmax(accuracies)],
        'best_accuracy': max(accuracies),
        'best_f1_dim': dimensions[np.argmax(f1_scores)],
        'best_f1': max(f1_scores),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
    }
    
    return {
        'per_dimension': metrics,
        'summary': summary,
    }

def plot_results(
        stats: Dict, 
        title: str = "Classification Performance vs Dimensions"
):
    """
    Plot the results taking stats and custom title as parameters
    """

    metrics = stats['per_dimension']
    dimensions = sorted(metrics.keys())
    
    accuracies = [metrics[d]['accuracy'] for d in dimensions]
    precisions = [metrics[d]['precision'] for d in dimensions]
    recalls = [metrics[d]['recall'] for d in dimensions]
    f1_scores = [metrics[d]['f1'] for d in dimensions]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot all metrics
    ax1 = axes[0]
    ax1.plot(dimensions, accuracies, 'o-', label='Accuracy', linewidth=2)
    ax1.plot(dimensions, precisions, 's-', label='Precision', linewidth=2)
    ax1.plot(dimensions, recalls, '^-', label='Recall', linewidth=2)
    ax1.plot(dimensions, f1_scores, 'd-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Number of Dimensions')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot accuracy and F1 with best dimension markers
    ax2 = axes[1]
    ax2.plot(dimensions, accuracies, 'o-', label='Accuracy', linewidth=2, color='tab:blue')
    ax2.plot(dimensions, f1_scores, 'd-', label='F1 Score', linewidth=2, color='tab:orange')
    
    # Mark best dimensions
    summary = stats['summary']
    ax2.axvline(x=summary['best_accuracy_dim'], color='tab:blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=summary['best_f1_dim'], color='tab:orange', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('Score')
    ax2.set_title(f"Best Acc: {summary['best_accuracy']:.3f} (d={summary['best_accuracy_dim']}), "
                  f"Best F1: {summary['best_f1']:.3f} (d={summary['best_f1_dim']})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig


def pipeline(
        data_name: str,
        dim_red_method: str,
        dimensions: List[int],
        classifier_type: str,
        output_dir: str = "results"
):
    """
    Input is data name, reduction method, dimensions array, and output value
    Loads the data, runs the classifier, performs data analysis, and plots
    Returns stats and figure
    """

    # directory for output
    os.makedirs(output_dir, exist_ok=True)
    
    # run classifier with dimensions reduced
    data = LOADERS.get(data_name, load_minst)()
    classifier_res = get_classifier_results(
        data=data,
        dim_red=dim_red_method,
        dimensions=dimensions,
        classifier_type=classifier_type
    )

    # analyze results (compare predictions to test labels)
    stats = analyze_data(
        classifier_res,
        data.y_test
    )

    figure = plot_results(
        stats=stats,
        title=f"{data_name} - {dim_red_method} - {classifier_type}"
    )
    
    # save figure
    base_name = f"{data_name}_{dim_red_method}_{classifier_type}"
    fig_path = os.path.join(output_dir, f"{base_name}.png")
    figure.savefig(fig_path, dpi=150, bbox_inches='tight')
    
    # save stats to JSON (convert numpy types to native Python)
    stats_serializable = json.loads(
        json.dumps(stats, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    )
    stats_path = os.path.join(output_dir, f"{base_name}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"Saved figure to {fig_path}")
    print(f"Saved stats to {stats_path}")
    
    return stats, figure


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run classifier experiments with dimensionality reduction")
    parser.add_argument("--data", type=str, default="faces", 
                        choices=list(LOADERS.keys()),
                        help="Dataset to use")
    parser.add_argument("--reducer", type=str, default="pca",
                        choices=list(REDUCERS.keys()),
                        help="Dimensionality reduction method")
    parser.add_argument("--classifier", type=str, default="knn",
                        choices=list(CLASSIFIERS.keys()),
                        help="Classifier type")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50],
                        help="Dimensions to test")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Running: {args.data} + {args.reducer} + {args.classifier}")
    print(f"Dimensions: {args.dims}")
    
    stats, fig = pipeline(
        data_name=args.data,
        dim_red_method=args.reducer,
        dimensions=args.dims,
        classifier_type=args.classifier,
        output_dir=args.output
    )
    
    print(f"\nBest accuracy: {stats['summary']['best_accuracy']:.3f} at d={stats['summary']['best_accuracy_dim']}")
    print(f"Best F1: {stats['summary']['best_f1']:.3f} at d={stats['summary']['best_f1_dim']}")

