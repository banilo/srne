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

# load HCP task data
print('Loading data...')
if on_server:
    X_task, labels = joblib.load('/storage/workspace/danilo/prni2015/preload_HT_3mm')
else:
    X_task, labels = joblib.load('/git/prni2015/preload_HT_3mm')

labels = np.int32(labels)
print('Done!')

##############################################################################
# define computation graph
##############################################################################

class SSEncoder(BaseEstimator):
    def __init__(self, gain1, learning_rate,
                 max_epochs=100, l1=0.1, l2=0.1):

        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)
        self.penalty_l1 = np.float32(l1)
        self.penalty_l2 = np.float32(l2)

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    # prox definitions from zermelozf
    def prox_l1(self, x, l=1.):
        """ Proximal operator of the l1 norm."""
        x_abs = T.abs_(x)
        return T.sgn(x) * (x_abs - l) * (x_abs > l)

    def prox_l2(self, x, l=1.):
        """ Proximal operator of the l2 norm."""
        return np.float32(1.) / (np.float32(1) + l) * x
    
    def prox_enet(self, x, l_l1, l_l2, t=1.):
        """Proximal operator for the elastic net at x"""
        x_abs = T.abs_(x)
        t = np.float32(t)
        prox_l1 = T.sgn(x) * (x_abs - t * l_l1) * (x_abs > t * l_l1)
        return prox_l1 / (np.float32(1.) + t * l_l2)

    def prox_group_l2(self, x, l):
        """proximity operator for l21 norm (L2 over columns and L1 over rows => one group per row)"""
        rows_norm = (x ** 2).sum(axis=1)
        shrink = T.maximum(1 - l / rows_norm, 0)
        return x * shrink[:, np.newaxis]

    # def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    #     grads = T.grad(cost=cost, wrt=params)
    #     updates = []
    #     for p, g in zip(params, grads):
    #         acc = theano.shared(p.get_value() * 0.)
    #         acc_new = rho * acc + (1 - rho) * g ** 2
    #         gradient_scaling = T.sqrt(acc_new + epsilon)
    #         g = g / gradient_scaling
    #         
    #         new_val = self.prox_enet(p - lr * g, self.penalty_l1, self.penalty_l2)
    #         
    #         updates.append((acc, acc_new))
    #         updates.append((p, new_val))
    #     return updates
        
    def ISTA(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            new_val = p - lr * g
            new_val = self.prox_group_l2(new_val, np.float32(0.1) * lr)
            new_val = self.prox_l1(new_val, np.float32(0.1) * lr)
            
            updates.append((p, new_val))
        return updates
        
    def get_param_pool(self):
        cur_params = (
            self.V0s
            # self.regcoeffs_col, self.V0s
        )
        return cur_params

    def fit(self, X_task, y, atlas_vox_reg):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.n_reg = atlas_vox_reg.shape[1]  # number of brain compartments
        n_input = X_task.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.input_taskdata = T.matrix(dtype='float32', name='input_taskdata')
        self.params_from_last_iters = []

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if not DEBUG_FLAG:
            X_train_s = theano.shared(
                value=np.float32(X_task), name='X_train_s')
            y_train_s = theano.shared(
                value=np.int32(y), name='y_train_s')
            lr_train_samples = len(X_task)
        else:
            from sklearn.cross_validation import StratifiedShuffleSplit
            folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20)
            new_trains, inds_val = iter(folder).next()
            X_train, X_val = X_task[new_trains], X_task[inds_val]
            y_train, y_val = y[new_trains], y[inds_val]

            X_train_s = theano.shared(value=np.float32(X_train),
                                      name='X_train_s', borrow=False)
            y_train_s = theano.shared(value=np.int32(y_train),
                                      name='y_train_s', borrow=False)
            # X_val_s = theano.shared(value=np.float32(X_val),
            #                         name='X_train_s', borrow=False)
            # y_val_s = theano.shared(value=np.int32(y_val),
            #                         name='y_cal_s', borrow=False)
            lr_train_samples = len(X_train)
            self.dbg_epochs_ = list()
            self.dbg_acc_train_ = list()
            self.dbg_acc_val_ = list()
            self.dbg_lr_cost_ = list()
            self.dbg_nonimprovesteps = list()
            self.dbg_prfs_ = list()
            self.dbg_prfs_other_ds_ = list()

        self.batch_size = X_train_s.shape[0]  # HACK

        # V -> supervised / logistic regression
        # regcoeff_vals = rng.randn(self.n_reg, self._nreg).astype(np.float32) * self.gain1        
        # self.regcoeffs_col = theano.shared(value=regcoeff_vals, name='regcoeffs_col')
        # regcoeffs = self.regcoeffs_col.repeat(self.n_reg, 1)  # copy into new columns
        # regcoeffs = (self.regcoeffs_col * T.ones((self.n_reg, self.n_reg))).T

        # computation graph: logistic regression
        clf_n_output = 18  # number of labels
        my_y = T.ivector(name='y')

        atlas_vox_reg = np.float32(atlas_vox_reg)
        self.altass_vox_reg = theano.shared(value=atlas_vox_reg, name='atlas')
        
        V0 = rng.randn(self.n_reg, clf_n_output).astype(np.float32) * self.gain1
        self.V0s = theano.shared(V0, name='V0')

        self.p_y_given_x = T.nnet.softmax(
            T.dot(T.dot(self.input_taskdata, self.altass_vox_reg), self.V0s)
            # T.dot(T.dot(T.dot(self.input_taskdata, self.altass_vox_reg), regcoeffs), self.V0s)
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        # self.lr_cost = (
        #     self.lr_cost
        #     # T.mean(abs(self.V0s)) * self.penalty_l1 +
        # 
        #     # T.mean((self.V0s ** np.float32(2))) * self.penalty_l2
        #     # T.mean((self.regcoeffs_col ** np.float32(2))) * self.penalty_l2
        # )

        # sparse group lasso penalty
        # for r in range(self.n_reg):
        #     # l2_term = 0.1 * T.sqrt(T.sum(self.V0s[r, :] ** np.float32(2)))
        #     l2_term = 0
        #     l1_term = 1.0 * T.sum(abs(self.V0s[r, :]))
        #     self.lr_cost += 0.1 * (l2_term + l1_term)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        # define graphs for update rules
        # my_params = [self.regcoeffs_col, self.V0s]
        my_params = [self.V0s]
        my_updates = self.ISTA(
            cost=self.lr_cost,
            params=my_params,
            lr=self.learning_rate)
        f_train = theano.function(
            [index],
            [self.lr_cost],
            givens=givens_lr,
            updates=my_updates, allow_input_downcast=True)

        # optimization loop
        start_time = time.time()
        lr_last_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            # print time estimate
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            lr_n_batches = lr_train_samples // self.batch_size

            # for i in range(lr_n_batches):

            args = f_train(0)  # HACK: full dataset!
            lr_cur_cost = args[0]

            # evaluate epoch cost
            if lr_last_cost < lr_cur_cost:
                no_improve_steps += 1
            else:
                lr_last_cost = lr_cur_cost
                no_improve_steps = 0

            # logistic
            acc_train = self.score(X_train, y_train)
            acc_val, prfs_val = self.score(X_val, y_val, return_prfs=True)
            
            cur_regcoeff = self.V0s.eval()
            n_wipedregs = (cur_regcoeff.ravel() == 0).sum()
            
            n_wipegroups = 0
            for iii in range(self.n_reg):
                n_wipegroups += np.sum(cur_regcoeff[iii, :]) == 0

            print('E:%i, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, reg-knocks:%i, group-knocks:%i, ae_badsteps:%i' % (
                i_epoch + 1, lr_cur_cost, acc_train, acc_val, n_wipedregs, n_wipegroups, no_improve_steps))

            # if (i_epoch % 10 == 0):
            self.dbg_lr_cost_.append(lr_cur_cost)
            self.dbg_epochs_.append(i_epoch + 1)
            self.dbg_nonimprovesteps.append(no_improve_steps)
            self.dbg_acc_train_.append(acc_train)
            self.dbg_acc_val_.append(acc_val)
            self.dbg_prfs_.append(prfs_val)
                
            # save parameters
            # print('Param pool!')
            param_pool = self.get_param_pool()
            self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_taskdata: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        del X_test_s
        del givens_te
        return predictions[0]

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc

##############################################################################
# computation
##############################################################################

for l1 in [0.5]:
    l2 = 0.999
    my_title = r'SRE: L1=%f L2=%f' % (l1, l2)
    print(my_title)
    estimator = SSEncoder(
        gain1=0.004,  # empirically determined by CV
        learning_rate = np.float32(0.0001),  # empirically determined by CV
        max_epochs=1000, l1=l1, l2=l2)
    
    estimator.fit(X_task, labels, FS_msdl)

    fname = my_title.replace(' ', '_').replace('+', '').replace(':', '').replace('__', '_').replace('%', '')
    cur_path = op.join(WRITE_DIR, fname)
    joblib.dump(estimator, cur_path)
    # estimator = joblib.load(cur_path)
    # plt.savefig(cur_path + '_SUMMARY.png', dpi=200)
    
    # dump data also as numpy array
    np.save(cur_path + 'dbg_epochs_', np.array(estimator.dbg_epochs_))
    np.save(cur_path + 'dbg_acc_train_', np.array(estimator.dbg_acc_train_))
    np.save(cur_path + 'dbg_acc_val_', np.array(estimator.dbg_acc_val_))
    np.save(cur_path + 'dbg_lr_cost_', np.array(estimator.dbg_lr_cost_))
    np.save(cur_path + 'dbg_nonimprovesteps', np.array(estimator.dbg_nonimprovesteps))
    np.save(cur_path + 'dbg_prfs_', np.array(estimator.dbg_prfs_))

