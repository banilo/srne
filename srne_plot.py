import os
import nibabel as nib
from nilearn import plotting
from matplotlib import pylab as plt
from nilearn.image import index_img

example_net_reg_priors = [
(0, 'Visual'),
(3, 'Auditory'),
(7, 'Left Frontoparietal'),
(11, 'Cerebellum')]

plt.close('all')
try:
    os.mkdir('plots')
except:
    pass

plotting.plot_anat('colin.nii', cut_coords=(0, 0, 0), draw_cross=False,
                  title='')
plt.savefig('plots/colin.png')

for i_prior, title_prior in example_net_reg_priors:
    out_fname = 'plots/prior_example_%s' % title_prior.replace(' ', '_')
    plotting.plot_epi('resources/dbg_raal.nii.gz', title='',
                      cut_coords=(0, 0, 0), draw_cross=False)
    plt.savefig(out_fname + '_reg.png')

    net_regdmn_nii = index_img('resources/rsn_12.nii.gz', i_prior)
    plotting.plot_stat_map(net_regdmn_nii, cut_coords=(0, 0, 0),
                           title='',
                           colorbar=False, draw_cross=False, black_bg=True)
    plt.savefig(out_fname + '_net.png')

    net_regdmn_nii = index_img('resources/rsn_12_assigned_aal.nii.gz', i_prior)
    plotting.plot_epi(net_regdmn_nii, cut_coords=(0, 0, 0), draw_cross=False,
                      title='')
    # plotting.plot_stat_map(stat_map_img=net_regdmn_nii,
    #                        bg_img=
    #                        cut_coords=(0, 0, 0),
    #                        title=title_prior,
    #                        colorbar=False, draw_cross=False, black_bg=True)
    plt.savefig(out_fname + '_regnet.png', transparent=True)