from data import(load_olivetti, load_newsgroups, load_fashion_minst, load_minst)

def classify_data(name):
    name = name.upper()

    if name == "OLIVETTI":
        data = load_olivetti()

    elif name == "NEWSGROUPS":
        data = load_newsgroups()

    elif name == "FASHION_MINST":
        data = load_fashion_minst()

    elif name == "MINST":
        data = load_minst()
    

    return data.X, data.y, data.n_samples, data.n_features

#test
"""
if __name__=="__main__":
    X,y,n_samples, n_features = classify_data("olivetti")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("n_samples:", n_samples)
    print("n_features:", n_features)
    """
