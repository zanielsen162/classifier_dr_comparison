from sklearn.datasets import fetch_openml, fetch_20newsgroups, fetch_olivetti_faces
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Load data functions return (X, y, n_samples, n_features)
- X: feature matrix (numpy array)
- y: labels
- n_samples: number of samples
- n_features: number of features
"""
def load_olivetti(center=True):
    X, y = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
    n_samples, n_features = X.shape

    if center:
        # Global centering (focus on one feature, centering all samples)
        X = X - X.mean(axis=0)
        # Local centering (focus on one sample, centering all features)
        X -= X.mean(axis=1).reshape(n_samples, -1)

    print(f"Olivetti: {n_samples} samples, {n_features} features")
    return X, y, n_samples, n_features

def load_newsgroups(max_features=5000):
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(newsgroups.data).toarray()  # convert sparse to dense
    y = newsgroups.target
    n_samples, n_features = X.shape

    print(f"Newsgroups: {n_samples} samples, {n_features} features")
    return X, y, n_samples, n_features

def load_fashion_minst():
    # Fashion-MNIST
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, parser='auto')
    n_samples, n_features = X.shape

    return X, y, n_samples, n_features

def load_minst():
    # MNIST
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser='auto')
    n_samples, n_features = X.shape

    return X, y, n_samples, n_features