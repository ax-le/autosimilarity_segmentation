# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:40:07 2022

@author: amarmore

Novelty cost, adapted from the work of Foote, see [1].
The kernel is of binary values : 1 and -1.
This code is deprecated, but could be used in comparison tests.

If interested, one should use the novelty version from the toolbox MSAF [2] instead,
whose parameters were tested and optimized.

References
----------
[1] J. Foote, "Automatic audio segmentation using a measure of audio novelty",
in: 2000 IEEE Int. Conf. Multimedia and Expo. ICME2000. Proc. Latest Advances 
in the Fast Changing World of Multimedia, vol. 1, IEEE, 2000,pp. 452–455.

[2] O. Nieto and J.P. Bello, "Systematic exploration of computational music
structure research.", in: ISMIR, 2016, pp. 547–553.
"""
# %% Using implementation from msaf
import as_seg.baseline_segmenter.lsd.msaf_copy as msaf_lsd
import numpy as np
from scipy import signal
from scipy.ndimage import filters

# Spectral Clustering Params
SCLUSTER_CONFIG = {
    "hier" : False, # Flat segmentation only

    # "num_layers" : 16, # only for hierarchical

    # params from M. Buisson https://github.com/morgan76/Triplet_Mining/blob/main/algorithms/scluster/config.py
    "scluster_k" : 9,
    "evec_smooth": 8,
    "rec_smooth" : 1,
    "rec_width"  : 1, 
}


def build_scluster_config(
    scluster_k=None,
    evec_smooth=None,
    rec_smooth=None,
    rec_width=None,
    config=None,
):
    """Build the LSD config in a step-by-step way.

    Priority order:
    1. start from SCLUSTER_CONFIG defaults,
    2. apply values from ``config`` if provided,
    3. apply explicit function arguments if they are not None.
    """
    final_config = dict(SCLUSTER_CONFIG)

    if config is not None:
        for key, value in config.items():
            final_config[key] = value

    if scluster_k is not None:
        final_config["scluster_k"] = scluster_k

    if evec_smooth is not None:
        final_config["evec_smooth"] = evec_smooth

    if rec_smooth is not None:
        final_config["rec_smooth"] = rec_smooth

    if rec_width is not None:
        final_config["rec_width"] = rec_width

    return final_config


def process_msaf_lsd(
    input_spectrogram,
    scluster_k=None,
    evec_smooth=None,
    rec_smooth=None,
    rec_width=None,
    config=None,
):
    """Main process.

    Parameters
    ----------
    input_spectrogram : np.ndarray
        Input feature matrix (bars × features).
    scluster_k : int or None, optional
        Number of clusters for spectral clustering.  When provided, overrides
        the value in SCLUSTER_CONFIG.  Default: None (use SCLUSTER_CONFIG).
    evec_smooth : int or None, optional
        Median-filter size applied to eigenvectors. When provided, overrides
        the value in SCLUSTER_CONFIG.
    rec_smooth : int or None, optional
        Median-filter size applied to the recurrence matrix. When provided,
        overrides the value in SCLUSTER_CONFIG.
    rec_width : int or None, optional
        Width used for the recurrence matrix. When provided, overrides the
        value in SCLUSTER_CONFIG.
    config : dict or None, optional
        Full LSD config override. Explicit keyword arguments take precedence
        over values from this dictionary.

    Returns
    -------
    est_idxs : np.array(N)
        Estimated indeces the segment boundaries in frames.
    est_labels : np.array(N-1)
        Estimated labels for the segments.
    """
    final_config = build_scluster_config(
        scluster_k=scluster_k,
        evec_smooth=evec_smooth,
        rec_smooth=rec_smooth,
        rec_width=rec_width,
        config=config,
    )

    input_shape = input_spectrogram.shape
    est_inter_list, est_labels_list, Cnorm = do_segmentation(
        input_spectrogram.T,
        config=final_config,
    )

    est_inter_list = np.array(est_inter_list, dtype=int)

    # Add first and last frames
    if 0 not in est_inter_list:
        est_inter_list = np.concatenate(([0], est_inter_list))
    if input_shape[0] - 1 not in est_inter_list:
        est_inter_list = np.concatenate((est_inter_list, [input_shape[0] - 1]))

    # Post process estimations
    est_idxs, est_labels = postprocess_msaf(est_inter_list, est_labels_list)

    return est_idxs, est_labels

def do_segmentation(input_spectrogram, config=SCLUSTER_CONFIG, in_bound_idxs=None):
    hier = config["hier"]
    #print('C shape',C.shape, flush=True)
    embedding = msaf_lsd.embed_beats(input_spectrogram, input_spectrogram, config=config)
    #print('embed shape',embedding.shape, flush=True)
    Cnorm = np.cumsum(embedding ** 2, axis=1) ** 0.5
    #print(Cnorm, flush=True)
    if hier:
        est_idxs = []
        est_labels = []
        for k in range(1, config["num_layers"] + 1):
            est_idx, est_label = msaf_lsd.cluster(embedding, Cnorm, k)
            #if select:
            est_idxs.append(est_idx)
            est_labels.append(np.asarray(est_label, dtype=int))

    else:
        est_idxs, est_labels = msaf_lsd.cluster(embedding, Cnorm, config["scluster_k"], in_bound_idxs)
        #print(len(est_idxs), flush=True)
        est_labels = np.asarray(est_labels, dtype=int)

    return est_idxs, est_labels, Cnorm
        
def postprocess_msaf(est_idxs, est_labels):
        """Post processes the estimations from the algorithm, removing empty
        segments and making sure the lenghts of the boundaries and labels
        match."""
        # Make sure we are using the previously input bounds, if any
        # if self.in_bound_idxs is not None:
            # F = self._preprocess()
            # est_labels = msaf.utils.synchronize_labels(self.in_bound_idxs, est_idxs,
                                              # est_labels, F.shape[0])
            # est_idxs = self.in_bound_idxs

        # Remove empty segments if needed
        est_idxs, est_labels = msaf_lsd.remove_empty_segments(est_idxs, est_labels)

        assert len(est_idxs) - 1 == len(est_labels), "Number of boundaries " \
            "(%d) and number of labels(%d) don't match" % (len(est_idxs),
                                                           len(est_labels))

        # Make sure the indeces are integers
        est_idxs = np.asarray(est_idxs, dtype=int)

        return est_idxs, est_labels
