from sklearn import cluster, decomposition
from sklearn.manifold import MDS, Isomap
from pydiffmap import diffusion_map as dm

class DataReducer:
    def __init__(self,
            components
    ):
        self.components = components


"""
Given the data name and the number of dimensions, returns the pca object
"""
def run_pca(data, num_dim=10):
    pca_estimator = decomposition.PCA(
        svd_solver="randomized", whiten=True, n_components=num_dim
    )
    transformed = pca_estimator.fit_transform(data.X)

    return DataReducer(transformed)

"""
Given the data name and number of dimensions, return diff map object
Note, using standard Gaussian kernel
"""
def run_diffusion_map(data, num_dim=10):
    diffusion_map = dm.DiffusionMap.from_sklearn(n_evecs=num_dim)
    diffusion_map.fit_transform(data.X)

    return DataReducer(diffusion_map.dmap)

"""
Given the data name and number of dimensions, return mds object
"""
def run_mds(data, num_dim=10):
    mds = MDS(n_components=num_dim, random_state=0)
    mds.fit_transform(data.X)

    return DataReducer(mds.embedding_)

"""
Given the data name and number of dimensions, return isomap object
"""
def run_isomap(data, num_dim=10):
    isomap = Isomap(n_components=num_dim)
    isomap.fit_transform(data.X)

    return DataReducer(isomap.embedding_)