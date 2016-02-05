import struct_estimator_dataratio_zerobrainlevel_weighted

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

n_classes = 18
plt.close('all')
REGS = ['tree-l2', 'l1']

READ_DIR = '/git/srne/srne_benchmark_dataratioFIXED100_zerobrainlevel_weighted/'
coeffs_ref = {}
for reg in REGS:
    coeffs_ref[reg] = []
    cur_dmp = op.join(READ_DIR, '%s_dataratio100_maxit100' % reg)
    print(cur_dmp)
    clf = joblib.load(cur_dmp)
    coef_ = np.array([est.W_.ravel() for est in clf.best_estimator_.estimators_]).T
    coeffs_ref[reg].append(coef_.ravel())
    coeffs_ref[reg] = np.array(coeffs_ref[reg]).squeeze()

READ_DIR = '/git/srne/srne_benchmark_dataratioFIXED20_zerobrainlevel_weighted/'
coeffs = {}
for reg in REGS:
    coeffs[reg] = []
    for i_fold in range(5):  # per fold
        cur_dmp = op.join(READ_DIR, '%s_dataratio20_maxit100' % reg)
        if i_fold > 0:
            cur_dmp += '_fold%i' % (i_fold + 1)
        print(cur_dmp)
        clf = joblib.load(cur_dmp)
        coef_ = np.array([est.W_.ravel() for est in clf.best_estimator_.estimators_]).T
        coeffs[reg].append(coef_.ravel())
    coeffs[reg] = np.array(coeffs[reg])

# coeffs_ss = coeffs.copy()
# for reg in REGS:
#     coeffs_ss[reg] = ss.fit_transform(coeffs_ss[reg].T).T

for reg in REGS:
    print('Variance of %s: %.2f' % (reg,
        np.abs(coeffs[reg]).var(axis=0).sum()))

def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    # if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
    
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    
    score = float(matches)/float(lena + lenb)
    return score

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

from itertools import combinations

# compare 20%-data model fits among each other
THRESH = 0.0001
measures = {}
for reg in REGS:
    print(reg)
    folder = combinations(range(5), 2)
    measures[reg] = []
    for i1, i2 in folder:
        a = coeffs[reg][i1]
        b = coeffs[reg][i2]
        
        a_bool = np.array(np.logical_or(a > THRESH, a < -THRESH), dtype=np.int)
        b_bool = np.array(np.logical_or(b > THRESH, b < -THRESH), dtype=np.int)
        
        measure = mutual_info_score(a_bool, b_bool)
        # measure = jaccard_similarity_score(a_bool, b_bool, normalize=False)
        # measure = adjusted_mutual_info_score(a_bool, b_bool)
        print('Metric: %.10f (non-zero a: %i + non-zero b: %i)' % (
            measure, a_bool.sum(), b_bool.sum()
        ))
        measures[reg].append(measure)
    print('Mean (Std): %f (+/- %f)' % (
        np.mean(measures[reg]), np.std(measures[reg])
    ))

# compare 20%-data model fits with 100%-data model fits
THRESH = 0.0001
for reg in REGS:
    for i_fold in range(5):
        print('%s / fold %i' % (reg, i_fold + 1))
        a = coeffs[reg][i_fold]
        # b = coeffs_ref[reg]
        
        a_bool = np.array(np.logical_or(a > THRESH, a < -THRESH), dtype=np.int)
        b_bool = np.array(np.logical_or(b > THRESH, b < -THRESH), dtype=np.int)
        
        # measure = mutual_info_score(a_bool, b_bool)
        # measure = jaccard_similarity_score(a_bool, b_bool, normalize=False)
        measure = adjusted_mutual_info_score(a_bool, b_bool)
        print('Metric: %.10f' % measure)
        
        
