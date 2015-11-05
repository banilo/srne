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
FS_msdl = nifti_masker.transform(r_msdl_nii).T
msdl_labels = np.argmax(FS_msdl, axis=1)
msdl_labels += 1 #  np.ones(len(FS_msdl))
n_regions = len(np.unique(msdl_labels))

assert n_regions == 39
nifti_masker.inverse_transform(msdl_labels).to_filename('dbg_msdl_labels.nii.gz')

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

print('Done!')

# prepare Smith2009 ICA components
from nilearn.image import resample_img
rsn_nii4d = nib.load('resources/rsn20.nii.gz')
aal_nii = nib.load('resources/aal.nii')
raal_nii = resample_img(
    aal_nii,
    nifti_masker.mask_img_.get_affine(),
    nifti_masker.mask_img_.shape,
    interpolation='nearest'
)
raal_nii.to_filename('resources/dbg_raal.nii.gz')
raal_data = nifti_masker.transform(raal_nii)[0]

my_rsns = index_img(
    rsn_nii4d,
    [5, 15, 9, 2, 1, 13, 10, 11, 12, 14, 7, 8])
my_rsns_data = nifti_masker.transform(my_rsns)


import spams
import scipy.sparse as ssp
from nilearn.image import index_img
from scipy.stats import zscore

class StructuredEstimator(BaseEstimator):
    def __init__(self, regul, lambda1):
        """
        Parameters
        ----------
        lambda : float
            Set importance of penalty term
            
        regul : string
            Choice of regularization

        Valid values for the regularization parameter (regul) are:
          "l1", "l2", "elastic-net",
          "sparse-group-lasso-l2", "sparse-group-lasso-linf",
          "tree-l0", "tree-l2", "trace-norm"
        """
        self.lambda1 = lambda1
        self.regul = regul
        
        net_reg_map = None

    def fit(self, X, y, reg_data=None, net_data=None):
        self.rng = np.random.RandomState(42)

        X = np.asfortranarray(X)
        Y = np.asfortranarray(Y[:, np.newaxis])
        W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64, order="FORTRAN")

        if 'tree' in self.regul and net_reg_map is not None:
        # define tree structure for SPAMS
        reg_labels = np.unique(reg_data)[1:]
        net_data = zscore(net_data, axis=1)  # no values from an RSN dominate
        combo_map = net_data.argmax(axis=0) + 1

        net_reg_map = np.zeros(net_data.shape)
        for reg_label in reg_labels:
            reg_inds = np.where(raal_data == reg_label)[0]
            rsn_assigns = np.argmax(net_data[:, reg_inds], axis=0)
            bins = np.bincount(rsn_assigns)
            i_assigned_rsn = np.argmax(bins)
            print('Region %i has %i voxels -> assigned to RSN index %i' %
                  (reg_label, len(reg_inds), i_assigned_rsn))
            print(bins)
            
            # write the current region to the assigned RSN
            net_reg_map[i_assigned_rsn, reg_inds] = reg_label

        net_reg_map_summed = np.sum(net_reg_map, axis=0)
        assert len(np.unique(net_reg_map_summed)) - 1 == 116  # each region has been assigned to a RSN?
        
        N_own_variables = []
        own_variables = []
        eta_g = np.array(np.ones(129),dtype=np.float32)
        groups = np.asfortranarray(np.zeros((129, 129)), dtype=np.bool)

        # add net info
        N_own_variables += list(np.zeros((13), dtype=np.int32))  # for root group + net groups
        own_variables = [np.int32(0)]

        groups[1:13, 0] = True  # each of the 12 nets belongs to root group

        cur_ind = 0
        for i_net in range(net_data.shape[0]):
            n_net_vox = np.count_nonzero(net_reg_map[i_net, :])
            own_variables += [np.int32(cur_ind)]
            cur_ind += n_net_vox  # move behind size of current net

        N_own_variables[0] = np.int32((net_reg_map_summed == 0).sum())  # zero entries in network label map belong to g1
        # all network sizes add up to the number of non-zero voxels in network label map?
        assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
        assert len(own_variables) == 13

        # add reg info
        X_task_tree = np.asfortranarray(np.zeros_like(X_task))

        cur_ind = 0
        i_gr = 13  # first group is root, then 12 net groups = 13 = [0..12]
        for i_net in range(net_data.shape[0]):
            regs_in_label = np.unique(net_reg_map[i_net, :])[1:]
            for reg_label in regs_in_label:
                reg_inds = np.where(net_reg_map[i_net, :] == reg_label)[0]
                n_reg_vox = len(reg_inds)
                own_variables += [np.int32(cur_ind)]
                N_own_variables += [n_reg_vox]  # no region voxels have decendences
                
                groups[i_gr, i_net + 1] = True  # cur reg belongs to cur net
                i_gr += 1
                
                X_task_tree[:, cur_ind:(cur_ind + n_reg_vox)] = X_task[:, reg_inds]

                cur_ind += n_reg_vox  # move behind size of current net

        # all region sizes add up to the number of non-zero voxels in network label map?
        assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
        assert groups.sum() - 12 == 116  # one dependence per region
        assert i_gr == 129
        assert len(N_own_variables) == 129

        # run SPAMS
        param = {'numThreads' : 1,'verbose' : True,
                 'lambda1' : 1.0, 'it0' : 10, 'max_it' : 200,
                 'L0' : 0.1, 'tol' : 1e-5, 'intercept' : False,
                 'pos' : False}
        tree = {
            'eta_g': np.float64(eta_g),
            'groups': groups,
            'own_variables': own_variables,
            'N_own_variables': N_own_variables}
        param['compute_gram'] = True
        param['regul'] = 'tree-l2'
        param['loss'] = 'logistic'

        (W, optim_info) = spams.fistaTree(
            np.float64(Y_train), np.float64(X_train), # U: double m x n matrix   (input signals) m is the signal size
            W0, tree, True,
            **param)

        if 'tree' in self.regul:
            # bring weight vector back in original order
            cur_ind = 0
            W_org = np.zeros_like(W)
            for i_net in range(my_rsns_data.shape[0]):
                regs_in_label = np.unique(net_reg_map[i_net, :])[1:]
                for reg_label in regs_in_label:
                    reg_inds = np.where(net_reg_map[i_net, :] == reg_label)[0]
                    n_reg_vox = len(reg_inds)
                    
                    W_org[reg_inds, 0] = W[cur_ind:(cur_ind + n_reg_vox), 0]

                    cur_ind += n_reg_vox  # move behind size of current net
            self.W_ = W_org
        else:
            self.W_ = W
        self.optim_info_ = optim_info

    def predict(self, X_test):
        y_pred = np.array(np.dot(X_test, self.W_) > 0,
                          dtype=np.float32)[:, 0]
        y_pred[y_pred == 0] = -1
        return y_pred

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        pred_y = np.array(pred_y, dtype=np.float32)[:, 0]
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc




# run SPAMs
own_variables =  np.array(own_variables, dtype=np.int32)
N_own_variables =  np.array(N_own_variables,dtype=np.int32)
groups = np.asfortranarray(groups)
groups = ssp.csc_matrix(groups, dtype=np.bool)
Y = np.asfortranarray(Y)
X_task_tree = np.asfortranarray(X_task_tree)
W0 = np.asfortranarray(np.float64(W0))

from sklearn.cross_validation import KFold
folder = KFold(len(Y), n_folds=10)
inds_train, inds_test = iter(folder).next()
X_train = X_task_tree[inds_train]
Y_train = Y[inds_train]
X_test = X_task_tree[inds_test]
Y_test = Y[inds_test]




out_fname = 'TOMaudio_vs_video_%s_lambda%.3f.nii.gz' % (
    param['regul'], param['lambda1']
)
nifti_masker.inverse_transform(W_org.T).to_filename(out_fname)

# rsync -vza dbzdok@drago:/storage/workspace/danilo/prni2015/TOM* /git/srne/

print('Accuracy: %.2f' % acc)


