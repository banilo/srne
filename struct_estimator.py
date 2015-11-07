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
import spams
import scipy.sparse as ssp
from nilearn.image import index_img
from scipy.stats import zscore

RES_NAME = 'sre'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

FORCE_TWO_CLASSES = False

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
msdl_labels = nifti_masker.transform(r_msdl_nii)[0, :]
msdl_labels += 1
n_regions = len(np.unique(msdl_labels))

assert n_regions == 117
nifti_masker.inverse_transform(msdl_labels[np.newaxis, :]).to_filename('dbg_msdl_labels.nii.gz')

# load HCP task data
print('Loading data...')
if on_server:
    X_task, labels = joblib.load('/storage/workspace/danilo/prni2015/preload_HT_3mm')
else:
    X_task, labels = joblib.load('/git/prni2015/preload_HT_3mm')

labels = np.array(labels)
if FORCE_TWO_CLASSES:
    inds1 = labels == 2  # TOM
    inds2 = labels == 3
    # inds1 = labels == 4  # object grasp/orientation
    # inds2 = labels == 5
    inds = np.logical_or(inds1, inds2)

    X_task = X_task[inds]
    Y = labels[inds].astype(np.float64)
    Y[Y==2] = -1
    Y[Y==3] = 1
else:
    Y = np.float64(labels)

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

class StructuredEstimator(BaseEstimator):
    def __init__(self, regul, lambda1, reg_data=None, net_data=None,
                 group_labels=None,
                 max_it=1000, n_threads=1, verbose=True):
        """
        Wrapper to enable access to SPAMS/Python-interface by
        a sklearn-typical estimator class.
        
        Parameters
        ----------
        lambda1 : float
            Set importance of penalty term
            
        regul : string
            Choice of regularization

        Valid values for the regularization parameter (regul) are:
          "l1", "l2",
          "sparse-group-lasso-l2", "sparse-group-lasso-linf",
          "tree-l0", "tree-l2", "trace-norm"
        """
        self.lambda1 = lambda1
        self.regul = regul
        self.reg_data = reg_data
        self.net_data = net_data
        self.group_labels = group_labels
        self.verbose = verbose
        self.max_it = max_it
        self.n_threads = n_threads
        
        self.net_reg_map = None

    def fit(self, X, y):
        if self.verbose:
            print(self)

        Y = np.asfortranarray(np.float64(y[:, np.newaxis]))
        W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64, order="FORTRAN")
        W0 = np.asfortranarray(np.float64(W0))

        if 'tree' in self.regul:
            X_task_tree = np.asfortranarray(np.zeros_like(X))
            if self.net_reg_map is None:
                # define tree structure for SPAMS
                reg_labels = np.unique(self.reg_data)[1:]
                self.net_data = zscore(self.net_data, axis=1)  # no values from an RSN dominate
                combo_map = self.net_data.argmax(axis=0) + 1

                self.net_reg_map = np.zeros(self.net_data.shape)
                for reg_label in reg_labels:
                    reg_inds = np.where(raal_data == reg_label)[0]
                    rsn_assigns = np.argmax(self.net_data[:, reg_inds], axis=0)
                    bins = np.bincount(rsn_assigns)
                    i_assigned_rsn = np.argmax(bins)
                    print('Region %i has %i voxels -> assigned to RSN index %i' %
                          (reg_label, len(reg_inds), i_assigned_rsn))
                    print(bins)
                    
                    # write the current region to the assigned RSN
                    self.net_reg_map[i_assigned_rsn, reg_inds] = reg_label

                net_reg_map_summed = np.sum(self.net_reg_map, axis=0)
                assert len(np.unique(net_reg_map_summed)) - 1 == 116  # each region has been assigned to a RSN?
                
                self.N_own_variables = []
                self.own_variables = []
                self.eta_g = np.array(np.ones(129),dtype=np.float32)
                self.groups = np.asfortranarray(np.zeros((129, 129)), dtype=np.bool)

                # add net info
                self.N_own_variables += list(np.zeros((13), dtype=np.int32))  # for root group + net groups
                self.own_variables = [np.int32(0)]

                self.groups[1:13, 0] = True  # each of the 12 nets belongs to root group

                cur_ind = 0
                for i_net in range(self.net_data.shape[0]):
                    n_net_vox = np.count_nonzero(self.net_reg_map[i_net, :])
                    self.own_variables += [np.int32(cur_ind)]
                    cur_ind += n_net_vox  # move behind size of current net

                self.N_own_variables[0] = np.int32((net_reg_map_summed == 0).sum())  # zero entries in network label map belong to g1
                # all network sizes add up to the number of non-zero voxels in network label map?
                assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
                assert len(self.own_variables) == 13

                # add reg info
                cur_ind = 0
                i_gr = 13  # first group is root, then 12 net groups = 13 = [0..12]
                for i_net in range(self.net_data.shape[0]):
                    regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                    for reg_label in regs_in_label:
                        reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                        n_reg_vox = len(reg_inds)
                        self.own_variables += [np.int32(cur_ind)]
                        self.N_own_variables += [n_reg_vox]  # no region voxels have decendences
                        
                        self.groups[i_gr, i_net + 1] = True  # cur reg belongs to cur net
                        i_gr += 1
                        
                        X_task_tree[:, cur_ind:(cur_ind + n_reg_vox)] = X[:, reg_inds]

                        cur_ind += n_reg_vox  # move behind size of current net

                # all region sizes add up to the number of non-zero voxels in network label map?
                assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
                assert self.groups.sum() - 12 == 116  # one dependence per region
                assert i_gr == 129
                assert len(self.N_own_variables) == 129
            
                self.own_variables =  np.array(self.own_variables, dtype=np.int32)
                self.N_own_variables =  np.array(self.N_own_variables,dtype=np.int32)
                self.groups = np.asfortranarray(self.groups)
                self.groups = ssp.csc_matrix(self.groups, dtype=np.bool)
            
            cur_ind = 0
            for i_net in range(self.net_data.shape[0]):
                regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                for reg_label in regs_in_label:
                    reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                    n_reg_vox = len(reg_inds)
                    
                    X_task_tree[:, cur_ind:(cur_ind + n_reg_vox)] = X[:, reg_inds]

                    cur_ind += n_reg_vox  # move behind size of current net
            X = X_task_tree

            # run SPAMS
            X = np.asfortranarray(X)

            param = {'numThreads' : self.n_threads,
                     'verbose' : self.verbose,
                     'lambda1' : float(self.lambda1),
                     'it0' : 10, 'max_it' : self.max_it,
                     'L0' : 0.1, 'tol' : 1e-05,
                     'intercept' : False,
                     'pos' : False}
            tree = {
                'eta_g': np.float64(self.eta_g),
                'groups': self.groups,
                'own_variables': self.own_variables,
                'N_own_variables': self.N_own_variables}
            param['compute_gram'] = True
            param['regul'] = self.regul
            param['loss'] = 'logistic'

            (W, optim_info) = spams.fistaTree(
                np.float64(Y), np.float64(X), # U: double m x n matrix   (input signals) m is the signal size
                W0, tree, True,
                **param)

            # bring weight vector back in original order
            cur_ind = 0
            W_org = np.zeros_like(W)
            for i_net in range(self.net_data.shape[0]):
                regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                for reg_label in regs_in_label:
                    reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                    n_reg_vox = len(reg_inds)
                    
                    W_org[reg_inds, 0] = W[cur_ind:(cur_ind + n_reg_vox), 0]

                    cur_ind += n_reg_vox  # move behind size of current net
            self.W_ = W_org
        # end if 'tree'
        else:
            X = np.asfortranarray(X)

            param = {'L0': 0.1,
             'a': 0.1,
             'b': 1000,
             'compute_gram': False,
             'intercept': False,
             'ista': False,
             'it0': 10,
             'lambda1': float(self.lambda1),
             'loss': 'logistic',
             'max_it': self.max_it,
             'numThreads': self.n_threads,
             'pos': False,
             'regul': self.regul,
             'subgrad': False,
             'tol': 1e-05,
             'verbose': self.verbose}
            if self.group_labels is not None:
                param['groups'] = np.int32(self.group_labels)
            (self.W_, optim_info) = spams.fistaFlat(Y, X, W0, True, **param)            
        
        self.optim_info_ = optim_info

    def decision_function(self, X):
        return np.array(np.dot(X, self.W_), dtype=np.float32)[:, 0]

    def predict(self, X_test):
        y_pred = np.array(np.dot(X_test, self.W_) > 0,
                          dtype=np.float32)[:, 0]
        y_pred[y_pred == 0] = -1
        return y_pred

    def score(self, X, y, return_prfs=False):
        pred_y = np.float32(self.predict(X))
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc




# run SPAMs
from sklearn.cross_validation import StratifiedShuffleSplit

folder = StratifiedShuffleSplit(Y, n_iter=10, test_size=0.1,
                                random_state=42)
inds_train, inds_test = iter(folder).next()
X_train = X_task[inds_train]
Y_train = Y[inds_train]
X_test = X_task[inds_test]
Y_test = Y[inds_test]

clf = StructuredEstimator(
    lambda1=0.25,
    regul='l2',
    reg_data=raal_data,
    net_data=my_rsns_data,
    max_it=2000,
    group_labels=None #np.int32(msdl_labels)
)


from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

param_grid = {'estimator__lambda1': np.float32(np.linspace(0, 0.5, 11))}
# param_grid = {'estimator__lambda1': [0.001, 0.25]}

clf_ovr = OneVsRestClassifier(clf, n_jobs=10)
clf_ovr_gs = GridSearchCV(clf_ovr, param_grid, n_jobs=1, cv=3)
clf_ovr_gs.fit(X_train, Y_train)

test_acc = clf_ovr_gs.score(X_test, Y_test)
print('Accuracy: %.2f' % test_acc)

# ovr_ests = clf_ovr_gs.best_estimator_



# test_acc = gs.best_estimator_.score(X_test, Y_test)
# print('Accuracy: %.2f' % test_acc)
# 
# 
# out_fname = 'test_TOMaudio_vs_video_%s_lambda%.3f.nii.gz' % (
#     clf.regul, clf.lambda1
# )
# nifti_masker.inverse_transform(clf.W_.T).to_filename(out_fname)

# rsync -vza dbzdok@drago:/storage/workspace/danilo/prni2015/TOM* /git/srne/



