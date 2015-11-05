"""
HCP: sparse region estimator

"One of the greatest challenges left for systems
neuroscience is to understand the normal and dysfunctional
operations of the cerebral cortex by relating local and global
patterns of activity [...]."

Buzsaki 2007 Nature
"""
print __doc__
"""
Notes:
working! with L1=0.1 wipes out ~300 of 702 variables w/o decrease
in out-of-sample performance
"""

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
from matplotlib import pylab as plt
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs, resample_img
import joblib
import time

RES_NAME = 'sre'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

on_server = op.exists(r'/storage')

##############################################################################
# load+preprocess data
##############################################################################

# load the mask
mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()
print('Mask shape is %i/%i/%i' % nifti_masker.mask_img_.get_data().shape)
print('Mask voxels: %i' % mask_nvox)

# load MSDL rois
import nilearn.datasets
msdl_path = nilearn.datasets.fetch_msdl_atlas()['maps']

msdl_path = 'resources/aal.nii'  # HACK !
msdl_nii = nib.load(msdl_path)

r_msdl_nii = resample_img(
    img=msdl_nii,
    target_affine=nifti_masker.mask_img_.get_affine(),
    target_shape=nifti_masker.mask_img_.shape,
    interpolation='nearest'
)
r_msdl_nii = nib.Nifti1Image(
    np.float32(r_msdl_nii.get_data()),
    r_msdl_nii.get_affine(),
    header = r_msdl_nii.get_header()
)
r_msdl_nii.to_filename('debug_rmsdl.nii.gz')
msdl_labels = nifti_masker.transform(r_msdl_nii)
msdl_labels += 1
n_regions = len(np.unique(msdl_labels))

assert n_regions == 117
nifti_masker.inverse_transform(msdl_labels).to_filename('dbg_msdl_labels.nii.gz')
msdl_labels = msdl_labels[0]

# load HCP task data
print('Loading data...')
if on_server:
    X_task, labels = joblib.load('/storage/workspace/danilo/prni2015/preload_HT_3mm')
else:
    X_task, labels = joblib.load('/git/prni2015/preload_HT_3mm')

labels = np.array(labels)
inds1 = labels == 2  # TOM
inds2 = labels == 3
# inds1 = labels == 4  # object grasp/orientation
# inds2 = labels == 5
inds = np.logical_or(inds1, inds2)

X_task = X_task[inds]
Y = labels[inds].astype(np.float64)
Y[Y==2] = -1
Y[Y==3] = 1

X = np.asfortranarray(X_task)
Y = np.asfortranarray(Y[:, np.newaxis])
W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64, order="FORTRAN")
print('Done!')

# prepare network dictionary
# spca = joblib.load('/git/cohort/archi/preload_compr_HTGSPCAModel40')
# from nilearn.input_data import NiftiMasker
# nm = NiftiMasker('/git/cohort/archi/debug_mask.nii')
# nm.fit()
# comps_nii = nm.inverse_transform(spca.components_)
# comps_nii.to_filename('spca_4d.nii.gz')
# comps_4d = nifti_masker.transform(comps_nii)
# 
# net_maxlabels = np.argmax(comps_4d, axis=0)  # HACK
# nifti_masker.inverse_transform(net_maxlabels).to_filename('dbg_net_maxlabels.nii.gz')
# msdl_labels = net_maxlabels
# 
# comps_4d[comps_4d < 0.1] = 0
# comps_4d[comps_4d >= 0.1] = 1
# n_networks = len(comps_4d)
# for i_comp in range(n_networks):
#     comps_4d[i_comp, :] *= (i_comp + 1 + n_regions)
# net_labels = nifti_masker.inverse_transform(comps_4d)
# net_labels.to_filename('dbg_net_labels.nii.gz')

import spams
param = {
 'L0': 0.1,
 'a': 0.1,
 'b': 1000,
 'compute_gram': False,
 'intercept': False,
 'ista': False,
 'it0': 10,
 'lambda1': 1.0,
 'loss': 'logistic',
 'max_it': 2000,
 'numThreads': 1,
 'pos': False,
 'regul': 'sparse-group-lasso-linf',
 'groups': np.int32(msdl_labels),
 'subgrad': False,
 'tol': 0.001,
 'verbose': True}
(W, optim_info) = spams.fistaFlat(
    Y, X, # U: double m x n matrix   (input signals) m is the signal size
    W0, True,
    **param)
print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))

out_fname = 'TOMaudio_vs_video_%s_lambda%.3f.nii.gz' % (
    param['regul'], param['lambda1']
)
nifti_masker.inverse_transform(W.T).to_filename(out_fname)

# rsync -vza dbzdok@drago:/storage/workspace/danilo/srne/TOM* /git/srne/



