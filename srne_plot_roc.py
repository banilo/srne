import struct_estimator_dataratio_zerobrainlevel_weighted

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import joblib

n_classes = 18

READ_DIR = 'srne_benchmark_dataratioFIXED100_zerobrainlevel_weighted'

REGS = ['tree-l2', 'l1',
        'elastic-net', 'trace-norm', 
        'sparse-group-lasso-l2', 'group-lasso-l2']
REGS_STR = ['Hierarchical Tree Sparsity', 'Lasso',
            'Elastic-Net', 'Trace-Norm', 
            'Sparse Group Lasso', 'Group Lasso']

auc_mat = np.zeros((len(REGS), n_classes))

for i_r, r in enumerate(REGS):
    print('-' * 80)
    print(r)
    dump_path = op.join(READ_DIR, '%s_dataratio100_maxit100' % r)
    clf = joblib.load(dump_path)

    print(len(clf.best_estimator_.estimators_))
    print('Mean Accuracy: %.3f' % np.mean(clf.test_acc))
    print('Mean Precision: %.3f' % np.mean(clf.test_prfs[0]))
    print('Mean Recall: %.3f' % np.mean(clf.test_prfs[1]))

    coef_ = np.array([est.W_.ravel() for
                      est in clf.best_estimator_.estimators_]).T

    y_score = np.dot(X_test, coef_)

    # Compute ROC curve and ROC area for each class

    Y_test_bin = label_binarize(Y_test, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_score[:, i])
        cur_auc = auc(fpr[i], tpr[i])
        roc_auc[i] = cur_auc
        auc_mat[i_r, i] = cur_auc

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        Y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: %s' % r)
    # plt.legend(loc="lower right")
    plt.show()
    plt.savefig(dump_path + '_ROC.png')

# summary AUC plot: per classifier
plt.figure()
for i_c in range(n_classes):
    plt.plot(np.arange(len(REGS)), auc_mat[:, i_c] * 100, linewidth=2)
plt.ylim([35., 102.])
plt.xticks(np.arange(len(REGS)), REGS, rotation=35)
plt.ylabel('AUC [%]')
plt.title('Comparing classifications of 18 tasks')
plt.tight_layout()
plt.show()
plt.savefig(op.join(READ_DIR, 'ROC_ALL_perclf.png'))

# summary AUC plot: per task
plt.figure()
for i_r in range(len(REGS)):
    plt.plot(np.arange(n_classes),
             auc_mat[i_r, :] * 100,
             label=REGS_STR[i_r],
             linewidth=2)
plt.ylim([35., 102.])
plt.xticks(np.arange(n_classes), np.arange(n_classes) + 1)
plt.ylabel('AUC [%]')
plt.title('Comparing classifications of 18 classes')
plt.xlabel('psychological task')
plt.tight_layout()
plt.legend(loc="middle right")
plt.show()
plt.savefig(op.join(READ_DIR, 'ROC_ALL_perclass.png'))
