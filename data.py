from sklearn.datasets import fetch_openml, fetch_20newsgroups, fetch_olivetti_faces
from sklearn.feature_extraction.text import TfidfVectorizer

# MNIST
X_mnist, y_mnist = fetch_openml("mnist_784", version=1, return_X_y=True, parser='auto')

# Fashion-MNIST
X_f_mnist, y_f_mnist = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, parser='auto')

# 20 Newsgroups (text data)
data = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(max_features=5000)
X_newsgroups = vectorizer.fit_transform(data.data)
y_newsgroups = data.target

# Olivetti Faces
data = fetch_olivetti_faces()
X_faces, y_faces = data.data, data.target