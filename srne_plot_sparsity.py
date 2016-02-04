# import struct_estimator_dataratio_zerobrainlevel_weighted

import os.path as op
import numpy as np
from scipy import linalg
import joblib
import nibabel as nib
from nilearn.input_data import NiftiMasker
from scipy.stats import pearsonr

n_classes = 18

def calc_sparsity(coef):
    # L1 norm divided by L2 norm of coef matrix (latter is Frobenius)
    return np.sum(np.abs(coef)) / linalg.norm(coef, ord='fro')

# load the class mean (=ground truth)
contrasts_names = [
    'REWARD-PUNISH', 'PUNISH-REWARD', 'SHAPES-FACES', 'FACES-SHAPES',
    'RANDOM-TOM', 'TOM-RANDOM',

    'MATH-STORY', 'STORY-MATH',
    'T-AVG', 'F-H', 'H-F',
    'MATCH-REL', 'REL-MATCH',

    'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG',
    '2BK-0BK'
]
mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()
task_means_z = np.zeros((mask_nvox, n_classes))
from scipy.stats import zscore
for i_c, c in enumerate(contrasts_names):
    cur_mean = '/git/cohort/archi/prelim2/compr2task_means/mean_%s.nii.gz' % c
    print cur_mean
    print op.exists(cur_mean)
    cur_task_mean = np.squeeze(nifti_masker.transform(cur_mean))
    cur_task_mean = np.nan_to_num(cur_task_mean)
    task_means_z[:, i_c] = cur_task_mean
    # task_means_z[:, i_c] = zscore(cur_task_mean)


REGS = ['tree-l2', 'l1']
dataratios = np.linspace(20, 100, 5)

for ratio in dataratios:
    READ_DIR = 'srne_benchmark_dataratioFIXED%i_zerobrainlevel_weighted'
    for reg in REGS:
        print('-' * 80)
        print('%s: %i' % (reg, ratio))

        dmppath = op.join(READ_DIR % ratio, '%s_dataratio%i_maxit100' % (reg, ratio))
        clf = joblib.load(dmppath)

        coef_ = np.array([est.W_.ravel() for est in clf.best_estimator_.estimators_]).T

        sparsity_all = calc_sparsity(coef_)
        print('Overall Sparsity mean: %.2f' % sparsity_all)

        sparsity_cl = np.ones(n_classes) * -1
        for i_cl in range(n_classes):
            sparsity_cl[i_cl] = calc_sparsity(coef_[:, i_cl][:, np.newaxis])
        print('Per-class Sparsity mean: %.2f' % np.mean(sparsity_cl))
        print('Per-class Sparsity std: %.2f' % np.std(sparsity_cl))
        
        
        coef_z = np.zeros_like(coef_)
        rhos = []
        for i_c in range(n_classes):
            coef_z[:, i_c] = zscore(coef_[:, i_c])
            coef_z[:, i_c] = np.nan_to_num(coef_z[:, i_c])
            
            # correlate class mean map (z scores) and z-scored model coeffs
            rho, p = pearsonr(coef_z[:, i_c], task_means_z[:, i_c])
            print('Support recovery (rho) for class %i: %.2f' % (
                i_c + 1, rho
            ))
            rhos.append(rho)
        print('Mean support recovery: %.2f +/- %.2f' % (
            np.mean(rhos), np.std(rhos))
        )
