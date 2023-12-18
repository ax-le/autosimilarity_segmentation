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
import msaf
import numpy as np
from scipy import signal
from scipy.ndimage import filters

def compute_foote_ssm(input_spectrogram, pre_filter = 12):
    spec = input_spectrogram
    if pre_filter != 0:
        median_filtering_size = pre_filter
        spec = median_filter(input_spectrogram, M=median_filtering_size)
        
    return msaf.algorithms.foote.segmenter.compute_ssm(spec)

def process_msaf_own_as(input_spectrogram = None, input_ssm = None, M_gaussian = 66, L_peaks = 64, pre_filter = 12, post_filter = 4):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated indeces the segment boundaries in frames.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        assert input_ssm is not None or input_spectrogram is not None, "Either input_ssm or input_spectrogram should be provided"

        # Make sure that the M_gaussian is even
        if M_gaussian % 2 == 1:
            M_gaussian += 1

        # Self similarity matrix
        if input_ssm is None:
            S = compute_foote_ssm(input_spectrogram, pre_filter)
            input_shape = input_spectrogram.shape
        else:
            S = input_ssm
            input_shape = input_ssm.shape

        # Compute gaussian kernel
        G = msaf.algorithms.foote.segmenter.compute_gaussian_krnl(M_gaussian)
        #plt.imshow(S, interpolation="nearest", aspect="auto"); plt.show()

        # Compute the novelty curve
        nc = msaf.algorithms.foote.segmenter.compute_nc(S, G)

        # Find peaks in the novelty curve
        sigma = post_filter
        est_idxs = pick_peaks(nc, sigma = sigma, L=L_peaks)

        # Add first and last frames
        est_idxs = np.concatenate(([0], est_idxs, [input_shape[0] - 1]))

        # Empty labels
        est_labels = np.ones(len(est_idxs) - 1) * -1

        # Post process estimations
        est_idxs, est_labels = postprocess_msaf(est_idxs, est_labels)

        return est_idxs, est_labels
        
def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X

def pick_peaks(nc, sigma = 4, L=16):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() / 20.

    if sigma != 0:
        nc = filters.gaussian_filter1d(nc, sigma=sigma)  # Smooth out nc

    th = filters.median_filter(nc, size=L) + offset
    #th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    #plt.plot(nc)
    #plt.plot(th)
    #for peak in peaks:
        #plt.axvline(peak)
    #plt.show()

    return peaks
        
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
        est_idxs, est_labels = msaf.utils.remove_empty_segments(est_idxs, est_labels)

        assert len(est_idxs) - 1 == len(est_labels), "Number of boundaries " \
            "(%d) and number of labels(%d) don't match" % (len(est_idxs),
                                                           len(est_labels))

        # Make sure the indeces are integers
        est_idxs = np.asarray(est_idxs, dtype=int)

        return est_idxs, est_labels

def my_process_segmentation_level(est_idxs, est_labels, N, frame_times, dur):
    """Processes a level of segmentation, and converts it into times.

    Parameters
    ----------
    est_idxs: np.array
        Estimated boundaries in frame indeces.
    est_labels: np.array
        Estimated labels.
    N: int
        Number of frames in the whole track.
    frame_times: np.array
        Time stamp for each frame.
    dur: float
        Duration of the audio track.

    Returns
    -------
    est_times: np.array
        Estimated segment boundaries in seconds.
    est_labels: np.array
        Estimated labels for each segment.
    """
    assert est_idxs[0] == 0 and est_idxs[-1] == N - 1
    assert len(est_idxs) - 1 == len(est_labels)

    # Add silences, if needed
    est_times = np.concatenate(([0], frame_times[est_idxs], [dur]))
    silence_label = np.max(est_labels) + 1
    est_labels = np.concatenate(([silence_label], est_labels, [silence_label]))

    # Remove empty segments if needed
    est_times, est_labels = msaf.utils.remove_empty_segments(est_times, est_labels)

    # I've added these lines because bar estimates may be less precise than frame and beats.
    if est_times[-1] < dur: # Fixing a bug, due to empty segment: [dur, dur]
        est_times.append(dur)
    if est_times[-1] > dur: # Fixing a bug, due to inaccuracies (rounding errors mainly)
        if est_times[-1] - dur < 1:
            est_times[-1] = dur
        else:
            #print(f"Ajout Axel: est_times[-1] = {est_times[-1]}, dur = {dur}, ce qui n'est pas normal. Voir si bug arrive souvent.")
            est_times[-1] = dur
    
    # Make sure that the first and last times are 0 and duration, respectively
    assert np.allclose([est_times[0]], [0]) and np.allclose([est_times[-1]], [dur])

    return est_times, est_labels


# %% Novelty computation
def novelty_cost(cropped_autosimilarity):
    """
    Novelty measure on this part of the autosimilarity matrix.
    The size of the kernel will be the size of the parameter matrix.

    Parameters
    ----------
    cropped_autosimilarity : list of list of floats or numpy array (matrix representation)
        The part of the autosimilarity which novelty measure is to compute.

    Raises
    ------
    NotImplementedError
        If the size of the autosimilarity is odd (novlety kernel can't fit this matrix).

    Returns
    -------
    float
        The novelty measure.

    """
    # Kernel is of the size of cropped_autosimilarity
    if len(cropped_autosimilarity) == 0:
        return 0
    
    if len(cropped_autosimilarity) % 2 == 1:
        raise NotImplementedError("The novelty computation is not implemented when the kernel is of odd size.") from None
        #return (novelty_cost(cropped_autosimilarity[:-1, :-1]) + novelty_cost(cropped_autosimilarity[1:, 1:])) / 2
    
    kernel_size = int(len(cropped_autosimilarity) / 2)
    kernel = np.kron(np.array([[1,-1], [-1, 1]]), np.ones((kernel_size, kernel_size)))
    return np.mean(kernel*cropped_autosimilarity)

def novelty_computation(autosimilarity_array, kernel_size):
    """
    Computes the novelty measure on the entire autosimilarity matrix, with a defined and fixed kernel size.

    Parameters
    ----------
    autosimilarity_array : list of list of floats or numpy array (matrix representation)
        The autosimilarity matrix.

    kernel_size : integer
        The size of the kernel.

    Raises
    ------
    NotImplementedError
        If the kernel size is odd, can't compute the novelty measure.

    Returns
    -------
    cost : list of float
        List of novelty measures, at each bar of the autosimilarity.

    """
    if kernel_size % 2 == 1:
        raise NotImplementedError("The novelty computation is not implemented when the kernel is of odd size.") from None
    cost = np.zeros(len(autosimilarity_array))
    half_kernel = int(kernel_size / 2)
    for i in range(half_kernel, len(autosimilarity_array) - half_kernel):
        cost[i] = novelty_cost(autosimilarity_array[i - half_kernel:i + half_kernel,i - half_kernel:i + half_kernel])
    return cost

# %% Post-processing of the novelty values
##################### Sandbox ##################
def peak_picking(tab, window_size = 1):
    """
    Returns the indexes of peaks of values in the given list of values.
    A value is considered "peak" if it's a local maximum,
    and if all values in the window (defined by 'window_size') before and after 
    are strictly monotonous.    
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The list of values to study.
    window_size : boolean, optional
        Size of the window around a possible peak to be considered "peak",
        ie number of consecutive values where the values should increase (before) and (decrease) after.
        The default is 1.

    Returns
    -------
    to_return : list of integers
        The indexes where values are peaking.

    """
    to_return = []
    for current_idx in range(window_size, len(tab) - window_size):
        if is_increasing(tab[current_idx - window_size:current_idx + 1]) and is_increasing(tab[current_idx:current_idx + window_size + 1][::-1]):
            to_return.append(current_idx)
    return to_return

def valley_picking(tab, window_size = 1):
    """
    Returns the indexes of valleys of values in the desired list of values.
    A value is considered "valley" if it's a local minimum,
    and if all values in the window (defined by 'window_size') before and after 
    are strictly monotonous.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The list of values to study.
    window_size : boolean, optional
        Size of the window around a possible valley to be considered "valley",
        ie number of consecutive values where the values should decrease (before) and increase (after).
        The default is 1.

    Returns
    -------
    to_return : list of integers
        The indexes where values are valleys.

    """
    to_return = []
    for current_idx in range(window_size, len(tab) - window_size):
        if is_increasing(tab[current_idx - window_size:current_idx + 1][::-1]) and is_increasing(tab[current_idx:current_idx + window_size + 1]):
            to_return.append(current_idx)
    return to_return

def is_increasing(tab):
    """
    Tests if the tab values are increasing.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The values.

    Returns
    -------
    boolean
        Whether the values are increasing or not.

    """
    if len(tab) <= 1 or len(np.unique(tab)) == 1:
        return False
    for idx in range(len(tab) - 1):
        if tab[idx] > tab[idx+1]:
            return False
    return True

def decreasing_peaks(data):
    """
    Returns the peaks indexes of a list of values in their decreasing order of values.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    data : list of float
        The values.

    Returns
    -------
    list of integers
        The indexes of the peaks, sorted in their decreasing order of values.

    """
    peaks_and_value = []
    for idx in peak_picking(data, window_size = 1):
        peaks_and_value.append((idx, data[idx]))
    return sorted(peaks_and_value, key=lambda x:x[1], reverse = True)

def select_highest_peaks_thresholded_indexes(data, percentage = 0.33):
    """
    Returns the peaks higher than a percentage of the maximal peak from a list of values.
    Used for peak picking in the novelty measure.
    
    Parameters
    ----------
    data : list of floats
        The values.
    percentage : float, optional
        The percentage of the maximal value for a peak to be valid.
        The default is 0.33.

    Returns
    -------
    list of integers
        Indexes of the valid peaks.

    """
    peaks = np.array(decreasing_peaks(data))
    max_peak = peaks[0,1]
    for idx, peak in enumerate(peaks):
        if peak[1] < percentage * max_peak:
            return [int(i) for i in sorted(peaks[:idx, 0])]
    return [int(i) for i in sorted(peaks[:,0])]

def values_as_slop(value, choice_func = max):
    """
    Compute peaks of a value (typically novelty measure)
    as the difference between absolute peaks and absolute valleys.
    Function choice_func determines the way of computing this gap.
    
    Typically, max will compute peaks as the maximum gap between a peaks and its two closest valleys,
    whereas min will select the minimal gap.
    
    This returns an array containing zeroes where there is no peak in absoluite value,
    and this new value as a gap computation where there was peaks before.

    Parameters
    ----------
    value : array of float
        The absolute value of the measure.
    choice_func : function name, optional
        Type of the function selecting the difference between peaks and valleys.
        Classical values are "max" for selecting the maximum gap between the peak and both its closest valleys,
        "min" for the minimum of both gaps, and "mean" (called as_seg.mean) for the mean of both gaps.
        The default is max.

    Returns
    -------
    peak_valley_slop : array of floats
        The new values of peaks as gaps, and 0 everywhere else.

    """
    peaks = peak_picking(value, window_size = 1)
    valleys = valley_picking(value, window_size = 1)
    peak_valley_slop = np.zeros(len(value))
    for peak in peaks:
        i = 0
        while i < len(valleys) and valleys[i] < peak:
            i+=1
        if i == 0:
            left_valley = 0
            right_valley = valleys[i]
        elif i == len(valleys):
            left_valley = valleys[i - 1]
            right_valley = 0
        else:
            left_valley = valleys[i - 1]
            right_valley = valleys[i]
        chosen_valley_value = choice_func(value[left_valley], value[right_valley])
        peak_valley_slop[peak] = value[peak] - chosen_valley_value
    return peak_valley_slop
    
def mean(val_a, val_b):
    """
    A function returning the mean of both values.
    This function is redeveloped so as to be called as choice_func in the function "values_as_slop()" (see above) in external projects.

    Parameters
    ----------
    val_a : float
        First value.
    val_b : float
        Second value.

    Returns
    -------
    float: mean of both values.

    """
    return (val_a + val_b) / 2