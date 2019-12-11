import numpy as np
from collections import defaultdict

NLCD_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_LC_CLASSES = 2#3

def get_nlcd_stats():
    cid = defaultdict(lambda: 0, {cl:i for i,cl in enumerate(NLCD_CLASSES)})
    nlcd_dist = np.zeros((len(NLCD_CLASSES), NUM_LC_CLASSES))
    nlcd_dist[:, :] = np.loadtxt('../data/nlcd_mu.txt')
    #nlcd_dist[:, :] = np.expand_dims(np.loadtxt('data/nlcd_mu.txt'),1)
    nlcd_var = np.zeros((len(NLCD_CLASSES), NUM_LC_CLASSES))
    #nlcd_var[:, :] = np.expand_dims(np.loadtxt('data/nlcd_sigma.txt'),1)
    nlcd_var[:, :] = np.loadtxt('../data/nlcd_sigma.txt')
    return cid, nlcd_dist, nlcd_var

