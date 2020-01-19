import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import os , shutil
import cv2 as cv 

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

    return X, filenames
X, filenames = load_features()
# Compute DBSCAN
eps=0.3
eps = 10
for eps in [0.01,0.1,0.3,1.,3.,10.,30.,100.]:
    db = DBSCAN(eps=eps, min_samples=3).fit(X)
    '''core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True'''

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # predict complete dataset
    #predicted = db.predict(X)
    #print('predicted:',predicted.shape,predicted.dtype,predicted.min(),predicted.max())

    print('labels',labels.shape,labels.min(),labels.max())

    '''print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))'''
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, labels))

    xy = X
    print('xy',xy.shape,xy.min(),xy.max())

    counts = {}
    clusters_dir = os.path.expanduser('~/data/cdli/symbol_clusters/eps_%f'%eps)
    for i in range(len(labels)):
        cluster_dir = os.path.join(clusters_dir,'%i'%(labels[i]))
        if not os.path.isdir(cluster_dir): os.makedirs(cluster_dir)

        if not labels[i] in counts: counts[labels[i]] = 0 
        pf = os.path.join(cluster_dir,'%i.png'%counts[labels[i]])
        shutil.copy(filenames[i],pf)
        counts[labels[i]] += 1

    for k in sorted(counts.keys()):
        print('cluster %i: %i'%(k,counts[k]))

    # make montage of cluster representatives
    grid_number = 1+int(np.sqrt(n_clusters_))
    grid_size = 128
    cc = 0
    mosaic = np.zeros((grid_number*grid_size,grid_number*grid_size,3),'uint8')
    for yi in range(grid_number):
        y = yi * grid_size
        for xi in range(grid_number):
            x = xi * grid_size
            czz = 0 
            stop = 0
            for zz in range(len(labels)):
                if labels[zz]==cc and not stop:
                    im = cv.imread(filenames[czz])
                    im = cv.resize(im,(grid_size,grid_size))
                    mosaic[y:y+grid_size,x:x+grid_size,:] = im 
                    cc += 1
                    stop=True
                czz += 1
    cv.imwrite(os.path.expanduser('~/data/cdli/symbol_clusters/mosaic_eps_%f.png'%eps),mosaic)