from sklearn import cluster, decomposition
from sklearn.manifold import MDS, Isomap
from pydiffmap import diffusion_map as dm
import numpy as np

class DataReducer:
    def __init__(self, train_components, test_components):
        self.train_components = train_components
        self.test_components = test_components


"""
Given the data name and the number of dimensions, returns the pca object
PCA supports proper fit on train, transform on test
"""
def run_pca(data, num_dim=10):
    pca_estimator = decomposition.PCA(
        svd_solver="randomized", whiten=True, n_components=num_dim
    )
    # Fit on train only, transform both
    train_transformed = pca_estimator.fit_transform(data.X_train)
    test_transformed = pca_estimator.transform(data.X_test)

    return DataReducer(train_transformed, test_transformed)

"""
Given the data name and number of dimensions, return diff map object
Note: Diffusion maps don't support out-of-sample extension easily,
so we fit on all data then split (unsupervised, so no label leakage)
"""
def run_diffusion_map(data, num_dim=10):
    diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=num_dim)
    # Fit on all data
    diffusion_map.fit_transform(data.X)
    
    # Split based on original indices
    n_train = len(data.X_train)
    train_components = diffusion_map.dmap[:n_train]
    test_components = diffusion_map.dmap[n_train:]

    return DataReducer(train_components, test_components)

"""
Given the data name and number of dimensions, return mds object
Note: MDS doesn't support transform, so we fit on all then split
"""
def run_mds(data, num_dim=10):
    mds = MDS(n_components=num_dim, random_state=0)
    # Stack train and test, fit on all
    X_all = np.vstack([data.X_train, data.X_test])
    mds.fit_transform(X_all)
    
    # Split based on train size
    n_train = len(data.X_train)
    train_components = mds.embedding_[:n_train]
    test_components = mds.embedding_[n_train:]

    return DataReducer(train_components, test_components)

"""
Given the data name and number of dimensions, return isomap object
Note: Isomap has transform but it can be unstable, so we fit on all then split
"""
def run_isomap(data, num_dim=10):
    isomap = Isomap(n_components=num_dim)
    # Stack train and test, fit on all
    X_all = np.vstack([data.X_train, data.X_test])
    isomap.fit_transform(X_all)
    
    # Split based on train size
    n_train = len(data.X_train)
    train_components = isomap.embedding_[:n_train]
    test_components = isomap.embedding_[n_train:]

    return DataReducer(train_components, test_components)