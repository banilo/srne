"""
HCP: sparse region estimator
"""
"""
Notes:
- 
"""

print __doc__

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

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
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
        self.batch_size = 100
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
        for r in range(self.n_reg):
            # l2_term = 0.1 * T.sqrt(T.sum(self.V0s[r, :] ** np.float32(2)))
            l2_term = 0
            l1_term = 1.0 * T.sum(abs(self.V0s[r, :]))
            self.lr_cost += 1 * (l2_term + l1_term)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        # define graphs for update rules
        # my_params = [self.regcoeffs_col, self.V0s]
        my_params = [self.V0s]
        my_updates = self.RMSprop(
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
            for i in range(lr_n_batches):
                args = f_train(i)
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
            
            cur_regcoeff = self.V0s.eval().ravel()
            n_wipedregs = (cur_regcoeff == 0).sum()

            print('E:%i, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, knockouts:%i, ae_badsteps:%i' % (
                i_epoch + 1, lr_cur_cost, acc_train, acc_val, n_wipedregs, no_improve_steps))

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

for l1 in [1]:
    l2 = 0.0
    my_title = r'SRE: L1=%.1f L2=%.1f' % (l1, l2)
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

STOP_CALCULATION

##############################################################################
# plot figures
##############################################################################

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%i' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            plot_stat_map(nii_z_fwhm, bg_img='colin.nii', threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))


# equally scaled plots
import re
pkgs = glob.glob(RES_NAME + '/*dbg_epochs*.npy')
dbg_epochs_ = np.load(pkgs[0])
dbg_epochs_ = np.load(pkgs[0])

d = {
    'training accuracy': '/*dbg_acc_train*.npy',
    'accuracy val': '/*dbg_acc_val_*.npy',
    'accuracy other ds': '/*dbg_acc_other_ds_*.npy',
    'loss ae': '/*dbg_ae_cost_*.npy',
    'loss lr': '/*dbg_lr_cost_*.npy',
    'loss combined': '/*dbg_combined_cost_*.npy'
}
n_comps = [20]

path_vanilla = 'nips3mm_vanilla'

for k, v in d.iteritems():
    pkgs = glob.glob(RES_NAME + v)
    for n_comp in n_comps:
        plt.figure()
        for p in pkgs:
            lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
            # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', p).group('comp'))
            n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
            if n_comp != n_hidden:
                continue
            
            dbg_acc_train_ = np.load(p)
            
            cur_label = 'n_comp=%i' % n_hidden
            cur_label += '/'
            cur_label += 'lambda=%.2f' % lambda_param
            cur_label += '/'
            if not '_AE' in p:
                cur_label += 'LR only!'
            elif 'subRS' in p:
                cur_label += 'RSnormal'
            elif 'spca20RS' in p:
                cur_label += 'RSspca20'
            elif 'pca20RS' in p:
                cur_label += 'RSpca20'
            cur_label += '/'
            cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
            cur_label += '' if '_AE' in p else '/LR only!'
            plt.plot(
                dbg_epochs_,
                dbg_acc_train_,
                label=cur_label)
        if k == 'training accuracy' or k == 'accuracy val':
            van_pkgs = glob.glob(path_vanilla + v)
            vanilla_values = np.load(van_pkgs[0])
            plt.plot(
                dbg_epochs_,
                vanilla_values,
                label='LR')
        plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
        plt.legend(loc='lower right', fontsize=9)
        plt.yticks(np.linspace(0., 1., 11))
        plt.ylabel(k)
        plt.xlabel('epochs')
        plt.ylim(0., 1.05)
        plt.grid(True)
        plt.show()
        plt.savefig(op.join(WRITE_DIR,
                    k.replace(' ', '_') + '_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_acc_val_*.npy')
for n_comp in n_comps:  # 
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', p).group('comp'))
        n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_acc_val_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_acc_val_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('validation set accuracy')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'accuracy_val_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_acc_other_ds_*.npy')
for n_comp in n_comps:  # 
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_acc_other_ds_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_acc_other_ds_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('ARCHI dataset accuracy')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'accuracy_archi_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_ae_cost_*.npy')
for n_comp in n_comps:  # AE
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_ae_cost_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_ae_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('AE loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_ae_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_lr_cost_*.npy')  # LR cost
for n_comp in n_comps:  # AE
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_lr_cost_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_lr_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('LR loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_lr_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_combined_cost_*.npy')  # combined loss
for n_comp in n_comps:  # AE
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_combined_cost_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_combined_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('combined loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_combined_%icomps.png' % n_comp))

# precision / recall / f1
target_lambda = 0.5

pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
        
        dbg_prfs_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_)[:, 0, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset precisions')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'prec_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# in-dataset recall at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
        
        dbg_prfs_ = np.load(p)
        
        dbg_prfs_ = np.load(p)
        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_)[:, 1, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset recall')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'rec_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# in-dataset f1 at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_ = np.load(p)
            
        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_)[:, 2, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset f1 score')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'f1_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset precision at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 0, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset precisions')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'prec_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset recall at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 1, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset recall')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'rec_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset f1 at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for p in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(p)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in p:
            cur_label += 'LR only!'
        elif 'subRS' in p:
            cur_label += 'RSnormal'
        elif 'pca20RS' in p:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 2, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset f1 score')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'f1_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# print network components (1st layer)
from nilearn.image import smooth_img
n_comp = 20
lmbd = 0.25
pkgs = glob.glob(RES_NAME + '/*W0comps.npy')
for p in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
        
    new_fname = 'comps_n=%i_lambda=%.2f_th0.0' % (n_hidden, lambda_param)
    comps = np.load(p)
    dump_comps(nifti_masker, new_fname, comps, threshold=0.0)

# print class weights
pkgs = glob.glob(RES_NAME + '/*W0comps.npy')
n_comp = 20
lmbd = 0.5
for p in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
    print p
    
    q = p.replace('W0', 'V1')
    comps = np.dot(np.load(q), np.load(p))
        
    new_fname = 'comps_n=%i_lambda=%.2f' % (n_hidden, lambda_param)
    dump_comps(nifti_masker, new_fname, comps, threshold=0.0, fwhm=None,
               perc=75)
    

# print LR decision matrix (2nd layer)
n_comp = 20
lmbd = 0.5
pkgs = glob.glob(RES_NAME + '/*V1comps.npy')
for p in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
    print p
    
    cur_mat = np.load(p)

    if n_comp == 20:
        fs = (8, 6)
    elif n_comp == 100:
        fs = (12, 8)
        

    plt.figure(figsize=fs)
    masked_data = np.ma.masked_where(cur_mat != 0., cur_mat)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.gray_r)
    masked_data = np.ma.masked_where(cur_mat == 0., cur_mat)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.show()

    # plt.xticks(range(n_comp)[::5], (np.arange(n_comp) + 1)[::5])
    # plt.xlabel('hidden components')
    # plt.yticks(range(18), np.arange(18) + 1)
    # plt.ylabel('tasks')
    # plt.title('Linear combinations of component per task')
    # plt.colorbar()
    
    new_fname = 'comps_n=%i_lambda=%.2f_V1_net2task.png' % (n_hidden, lambda_param)
    plt.savefig(op.join(WRITE_DIR, new_fname))

# out-of-dataset f1 score summary plots
for n_comp in [20, 50, 100]:
    f1_mean_per_lambda = list()
    f1_std_per_lambda = list()
    lambs = [0.25, 0.5, 0.75, 1.0]
    for target_lambda in lambs:
        pkgs = glob.glob(RES_NAME + '/*n_comp=%i*lambda=%.2f*dbg_prfs_other_ds_.npy' %
            (n_comp, target_lambda))
        print pkgs
        dbg_prfs_other_ds_ = np.load(pkgs[0])
        cur_mean = np.mean(dbg_prfs_other_ds_[-1, 2, :])
        f1_mean_per_lambda.append(cur_mean)
        cur_std = np.std(dbg_prfs_other_ds_[-1, 2, :])
        f1_std_per_lambda.append(cur_std)
        print('F1 means: %.2f +/- %.2f (SD)' % (cur_mean, cur_std))

    f1_mean_per_lambda = np.array(f1_mean_per_lambda)
    f1_std_per_lambda = np.array(f1_std_per_lambda)

    plt.figure()
    ind = np.arange(4)
    width = 1.
    colors = [#(7., 116., 242.), #(7., 176., 242.)
        #(7., 136., 217.), (7., 40., 164.), (1., 4., 64.)]
        (7., 176., 242.), (7., 136., 217.), (7., 40., 164.), (1., 4., 64.)]
    my_colors = [(x/256, y/256, z/256) for x, y, z in colors]
    plt.bar(ind, f1_mean_per_lambda, yerr=f1_std_per_lambda,
            width=width, color=my_colors)
    plt.ylabel('mean F1 score (+/- SD)')
    plt.title('out-of-dataset performance\n'
              '%i components' % n_comp)
    tick_strs = [u'low-rank $\lambda=%.2f$' % val for val in lambs]
    plt.xticks(ind + width / 2., tick_strs, rotation=320)
    plt.ylim(.5, 1.0)
    plt.grid(True)
    plt.yticks(np.linspace(0.5, 1., 11), np.linspace(0.5, 1., 11))
    plt.tight_layout()
    out_path2 = op.join(WRITE_DIR, 'f1_bars_comp=%i.png' % n_comp)
    plt.savefig(out_path2)
