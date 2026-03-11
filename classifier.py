import numpy as numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def KNN(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def logreg(X_train, y_train, X_test):
    logreg = LogisticRegression(solver='lbfgs', max_iter=5000)
    logreg.fit(X_train, y_train)
    return logreg.predict(X_test)


def naive_bayes(X_train, y_train, X_test):
    naive = GaussianNB()
    naive.fit(X_train, y_train)
    return naive.predict(X_test)