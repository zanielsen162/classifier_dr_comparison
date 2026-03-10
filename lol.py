import numpy as numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


from data import load_olivetti, load_newsgroups, load_fashion_minst, load_minst

def load_data(name):
    name = name.upper()

    if name == "OLIVETTI":
        data = load_olivetti()

    elif name == "NEWSGROUPS":
        data = load_newsgroups()

    elif name == "FASHION_MINST":
        data = load_fashion_minst()

    elif name == "MINST":
        data = load_minst()
    
    return data

def KNN_traindata(name):
    data = load_data(name)
    kkn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(data.X, data.y)
    return knn.predict(data.X)


def logreg_traindata(name):
    data = load_data(name)
    logreg = LogisticRegression(solver = 'liblinear', max_iter=1000)
    logreg.fit(data.X, data.y)
    return logreg.predict(data.X)


def naive_traindata(name):
    data = load_data(name)
    naive = GaussianNB()
    naive.fit(data.X, data.y)
    return naive.predict(data.X)




#data.py passes me the data
#this file should classify the data by matching the data in data.py to the format used for the true labels so we can compare the labels in the next file
#should output something that matches the format of data.y

#train each of the following data sets on the original features ^^

#k-NN
#logistic regression
#naive bayes