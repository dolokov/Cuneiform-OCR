import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def load_features(file_encodings='single_symbol_encodings.csv'):
    with open(file_encodings,'r') as f:
        lines = f.readlines()
        lines = [l.replace('\n','') for l in lines]

        filenames = [l.split(',')[0] for l in lines]
        features = [l.split(',')[1:] for l in lines]
        X = np.float32(np.array(features))
        print('features',X.shape,'minmax',X.min(),X.max(),'mean/std',X.mean(),X.std())
        X = StandardScaler().fit_transform(X)
        print('scaled',X.shape,'minmax',X.min(),X.max(),'mean/std',X.mean(),X.std())

    return X
X = load_features()
# Compute DBSCAN
eps=0.3
eps = 10
db = DBSCAN(eps=eps, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
'''print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))'''
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))

xy = X[core_samples_mask]
print('xy',xy.shape,xy.min(),xy.max())
for i in range(n_clusters_):
    ''