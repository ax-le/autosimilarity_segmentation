# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:42 2020

@author: amarmore

Correlation "Block-Matching" (CBM) algorithm.
This algorithm is designed to segment barwise autosimilarities.

In short, this algorithm focuses on the criteria of homogeneity to estimate segments, 
and computes an optimal segmentation via dynamic programming.

See [1] for more details.

References
----------
[1] Marmoret, A., Cohen, J. E., & Bimbot, F. (2023). Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 6(1), 167-185.
"""

import as_seg.model.errors as err

import math
import numpy as np
from scipy.sparse import diags
from scipy.signal import fftconvolve
import warnings

def compute_cbm(autosimilarity, min_size = 1, max_size = 32, penalty_weight = 1, penalty_func = "modulo8", bands_number = None, compute_score_using_convolve = False):
    """
    Dynamic programming algorithm, maximizing an overall score at the song scale, sum of all segments' scores on the autosimilarity.
    Each segment' score is a combination of
     - the correlation score on this the segment, dependong on the kernel, 
     - a penalty cost, function of the size of the segment, to enforce specific sizes (with prior knowledge),

    The penalty cost is computed in the function "penalty_cost_from_arg()".
    See this function for further details.

    It returns the optimal segmentation according to this score.
    
    This algortihm is also described in [1].
    
    IDEAS FOR FUTURE DEVELOPMENT: 
        - May be optimized using scipy.signal.fftconvolve(), but requires a different approach in parsing all segments.
    (i.e. parsing all possible segments for a specified kernel size and then retrieve the indexes of the diagonally-centered values)
            -- DONE, but not faster. Useless.
        - Taking into account values which are not around the diagonal but everywhere in the matrix in order to account for the repetition principle.

    Parameters
    ----------
    autosimilarity : list of list of float (list of columns)
        The autosimilarity to segment.
    min_size : integer, optional
        The minimal length of segments.
        The default is 1.
    max_size : integer, optional
        The maximal length of segments.
        The default is 32.
    penalty_weight : float, optional
        The ponderation parameter for the penalty function
    penalty_func : string
        The type of penalty function to use.
        See "penalty_cost_from_arg()" for further details.
    bands_number : positive integer or None, optional
        The number of bands in the kernel. 
        For the full kernel, bands_number must be set to None
        (or higher than the maximal size, but cumbersome)
        See [1] for details. 
        The default is None.
    compute_score_using_convolve : boolean, optional
        Whether to compute the correlation score using fftconvolve.
        Expected to be faster, but is actually not, or not that much.
        May still be useful for taking into account values which are not around the diagonal 
        but everywhere in the matrix in order to account for the repetition principle.
        The default is False.

    Raises
    ------
    ToDebugException
        If the program fails, generally meaning that the autosimilarity is incorrect.

    Returns
    -------
    list of tuples
        The segments, as a list of tuples (start, end).
    integer
        Global score (the maximum among all).
    """
    if compute_score_using_convolve:
        warnings.warn("compute_score_using_convolve is set to True, but it is not that faster, so it is highly DEPRECATED. May still be useful for future developments in order to take into account values which are not around the diagonal but everywhere in the matrix in order to account for the repetition principle.")
    
    N = len(autosimilarity)

    # Initialization of the scores and segments_best_starts.
    scores = [-math.inf for i in range(N+1)]
    scores[0] = 0

    # Initialize the best start index for each segment.
    segments_best_starts = [None for i in range(N+1)]
    segments_best_starts[0] = 0

    # Precompute all kernels
    kernels = compute_all_kernels(max_size, bands_number=bands_number)

    if compute_score_using_convolve:
        # Precompute the correlation scores for all possible segments using fft.convolve.
        precomputed_scores, max_corr_seg_size_eight = precompute_corr_scores_conv(autosimilarity=autosimilarity, kernels=kernels, min_size=min_size, max_size=max_size, len_autosimilarity=N, normalization_type = 'squared_length')
    else:
        precomputed_scores = None
        max_corr_seg_size_eight = None

    # Compute the maximal score value for size eight segments, if not already computed (whether failed in fast_mode or no fast_mode)
    if max_corr_seg_size_eight is None:
        max_corr_seg_size_eight = np.amax(correlation_entire_matrix_computation(autosimilarity, kernels, kernel_size = 8))

    # Main loop, computing scores for all possible segments.
    for current_idx in range(1, N+1): # Parse all indexes of the autosimilarity. Includes the last index N.
        # Looping on all possible start indexes for the current end index.
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index, shouldn't happen.") from None
            
            # Length of the segment
            segment_length = current_idx - possible_start_idx

            # Compute correlation score
            if precomputed_scores is None: # If not computed (original mode, correlation_score is computed at each step)
                corr_score = correlation_score(autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx], kernels, normalization_type = 'squared_length')
            else: # If already computed (fast mode, fft.convolve)
                # Retrieve precomputed correlation score
                # Extra check that valid segment length and start index
                if segment_length < precomputed_scores.shape[1] and possible_start_idx < precomputed_scores.shape[0]:
                    corr_score = precomputed_scores[possible_start_idx, segment_length]
                else:
                    raise err.ToDebugException("Invalid value of segment length or start index, shouldn't happen.") from None

            if corr_score == -np.inf:
                raise err.ToDebugException("Invalid value of correlation score, shouldn't happen.") from None

            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            this_segment_score = corr_score * segment_length - penalty_cost * penalty_weight * max_corr_seg_size_eight

            if possible_start_idx == 0: # Avoiding errors, as scores values are initially set to -inf.
                if this_segment_score > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = this_segment_score
                    segments_best_starts[current_idx] = 0
            else:
                if scores[possible_start_idx] + this_segment_score > scores[current_idx]: # This segment is of larger score
                    scores[current_idx] = scores[possible_start_idx] + this_segment_score
                    segments_best_starts[current_idx] = possible_start_idx

    segments = backtracking_for_best_segments(segments_best_starts, len_autosimilarity=N)
    return segments, scores[-1]

def precompute_corr_scores_conv(autosimilarity, kernels, min_size, max_size, len_autosimilarity, normalization_type = 'squared_length'):
    """
    Precompute the correlation scores for all possible segments using fft.convolve.
    Assumed to be faster than computing the cost for each segment in the main for loops of the CBM algorithm.

    Parameters
    ----------
    autosimilarity : np.ndarray
        The autosimilarity matrix.
    kernels : list of np.ndarray
        The kernels to use for the convolution.
    min_size : int
        The minimum size of the segments.
    max_size : int
        The maximum size of the segments.
    len_autosimilarity : int
        The length of the autosimilarity matrix.

    Returns
    -------
    np.ndarray
        The precomputed correlation scores.
    np.ndarray
        The maximum correlation score for kernel size 8.
    """
    conv_computed_scores = None
    max_corr_seg_size_eight = None

    # The condition on max_size < N is because the conv algo actually fails if max_size is too large, and is faster not using convs.
    if max_size < len_autosimilarity: 
        # Precompute the costs for all possible segments using convolution on the whole matrix. 
        # conv_computed_scores[s, l] will store cost of segment starting at index s with length l.
        conv_computed_scores = np.full((len_autosimilarity, max_size + 1), -np.inf)

        for p in range(min_size, max_size + 1):
            # Convolve autosimilarity with kernel
            # mode='valid' returns parts where kernel and signal fully overlap
            # Result size is (N-p+1, N-p+1)
            conv_res = fftconvolve(autosimilarity, kernels[p], mode='valid')
            
            # We are only interested in segments on the diagonal
            # conv_res[k, k] corresponds to the block autosimilarity[k:k+p, k:k+p]
            # diagonal contains costs for segments starting at index k (0 to N-p)
            diag_costs = conv_res.diagonal()
            
            # Normalize by the length of the segment, squared. May change. (ex: number of nonzero elements)
            diag_costs = normalize_correlation_measure(diag_costs, p, normalization_type = normalization_type)
            
            # Store in our lookup table
            # We assign to column p, rows 0 to len(diag_costs)
            conv_computed_scores[:len(diag_costs), p] = diag_costs
            
            if p == 8: # Normalization constant
                max_corr_seg_size_eight = np.amax(diag_costs)

    return conv_computed_scores, max_corr_seg_size_eight


def backtracking_for_best_segments(segments_best_starts, len_autosimilarity):
    """
    Finds the best segments using the computed best start for each segment.
    
    Parameters
    ----------
    segments_best_starts : list
        The best starts of the segments.
    len_autosimilarity : int
        The length of the autosimilarity.

    Returns
    -------
    segments : list
        The best segments.
    """
    # Because a segment's start is the previous one's end.
    precedent_frontier = segments_best_starts[len_autosimilarity] # Check that it does not break everything!
    # Initialize with the last segment.
    segments = [(precedent_frontier, len_autosimilarity)] # Segments are (start, end).

    # While we haven't reached the beginning of the autosimilarity.
    while precedent_frontier > 0: 
        # Add the segment to the list.
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        # Update the precedent frontier.
        precedent_frontier = segments_best_starts[precedent_frontier] 
        # If the precedent frontier is None, it means that the dynamic programming algorithm took an impossible path.
        if precedent_frontier == None: 
            raise err.ToDebugException("Well... The dynamic programming algorithm took an impossible path, so it failed. Understand why.") from None

    # Reverse the list to get the segments in the correct order.
    return segments[::-1] 

def compute_all_kernels(max_size, bands_number = None):
    """
    Precomputes all kernels of size 0 ([0]) to max_size, to be reused in the CBM algorithm.
    
    This is used for acceleration purposes.

    Parameters
    ----------
    max_size : integer
        The maximal size (included) for kernels.
    bands_number : positive integer or None, optional
        The number of bands in the kernel. 
        For the full kernel, bands_number must be set to None
        (or higher than the maximal size, but cumbersome)
        See [1] for details. 
        The default is None.
        
    Returns
    -------
    kernels : array of arrays (which are kernels)
        All the kernels, of size 0 ([0]) to max_size.

    """
    kernels = [[0]]
    for p in range(1,max_size + 1):
        if bands_number is None or p < bands_number:
            kern = np.ones((p,p)) - np.identity(p)
        else:
            k = np.array([np.ones(p-i) for i in np.abs(range(-bands_number, bands_number + 1))],dtype=object)
            offset = [i for i in range(-bands_number, bands_number + 1)]
            kern = diags(k,offset).toarray() - np.identity(p)
        kernels.append(kern)
    return kernels

def correlation_score(cropped_autosimilarity, kernels, normalization_type = 'squared_length'):
    """
    The correlation score on this part of the autosimilarity matrix.

    Parameters
    ----------
    cropped_autosimilarity : list of list of floats or numpy array (matrix representation)
        The part of the autosimilarity which correlation score is to compute.
    kernels : list of arrays
        Acceptable kernels.

    Returns
    -------
    float
        The correlation score.

    """
    p = len(cropped_autosimilarity)
    kern = kernels[p]
    correlation_measure = np.sum(np.multiply(kern,cropped_autosimilarity))
    return normalize_correlation_measure(correlation_measure, p, normalization_type)

def normalize_correlation_measure(correlation_measure, kernel_size, normalization_type = 'squared_length'):
    """
    Normalizes the correlation measure by the kernel size.
    
    Parameters
    ----------
    correlation_measure : float
        The correlation measure.
    kernel_size : integer
        The size of the kernel.

    Returns
    -------
    float
        The normalized correlation measure.
    """
    match normalization_type:
        case 'squared_length':
            return correlation_measure / kernel_size**2
        case 'length':
            return correlation_measure / kernel_size
        case 'number_of_nonzero_elements':
            return correlation_measure / np.count_nonzero(kernels[kernel_size])
        case 'none':
            return correlation_measure
        case _:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
    
def correlation_entire_matrix_computation(autosimilarity_array, kernels, kernel_size = 8):
    """
    Computes the correlation measure on the entire autosimilarity matrix, with a defined and fixed kernel size.
    Used for normalization purposes.
    
    Parameters
    ----------
    autosimilarity_array : list of list of floats or numpy array (matrix representation)
        The autosimilarity matrix.
    kernels : list of arrays
        All acceptable kernels.
    kernel_size : integer
        The size of the kernel for this measure.

    Returns
    -------
    cost : list of float
        List of correlation measures, at each bar of the autosimilarity.

    """
    cost = np.zeros(len(autosimilarity_array))
    for i in range(kernel_size, len(autosimilarity_array)):
        cost[i] = correlation_score(autosimilarity_array[i - kernel_size:i,i - kernel_size:i], kernels)
    return cost

def penalty_cost_from_arg(penalty_func, segment_length):
    """
    Returns a penalty cost, function of the size of the segment.
    The penalty function has to be specified, and is bound to evolve in the near future,
    so this docstring won't explain it.
    Instead, you'll have to read the code, sorry! It is pretty straightforward though.
    
    The ``modulo'' functions are based on empirical prior knowledge,
    following the fact that pop music is generally composed of segments of 4 or 8 bars.
    Still, penalty values are empirically set.

    Parameters
    ----------
    penalty_func : string
        Identifier of the penalty function.
    segment_length : integer
        Size of the segment.

    Returns
    -------
    float
        The penalty cost.

    """
    if penalty_func == "modulo8":        
        if segment_length == 8:
            return 0
        elif segment_length %4 == 0:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "modulo4":
        if segment_length %4 == 0:
            return 0
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "modulo8modulo4":        
        if segment_length == 8:
            return 0
        elif segment_length == 4:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "target_deviation_8_alpha_half": 
         return abs(segment_length - 8) ** (1/2)
    if penalty_func == "target_deviation_8_alpha_one": 
         return abs(segment_length - 8)
    if penalty_func == "target_deviation_8_alpha_two": 
         return abs(segment_length - 8) ** 2
    else:
        raise err.InvalidArgumentValueException(f"Penalty function not understood {penalty_func}.")

def possible_segment_start(idx, min_size = 1, max_size = None):
    """
    Generates the list of all possible starts of segments given the index of its end.
    
    Parameters
    ----------
    idx: integer
        The end of a segment.
    min_size: integer
        Minimal length of a segment.
    max_size: integer
        Maximal length of a segment.
        
    Returns
    -------
    list of integers
        All potentials starts of structural segments.
    """
    if min_size < 1: # No segment should be allowed to be 0 size
        raise err.InvalidArgumentValueException(f"Invalid minimal size: {min_size} (No segment should be allowed to be 0 or negative size).")
        #min_size = 1
    if max_size == None:
        return range(0, idx - min_size + 1)
    else:
        if idx >= max_size:
            return range(idx - max_size, idx - min_size + 1)
        elif idx >= min_size:
            return range(0, idx - min_size + 1)
        else:
            return []
        
# %% Scikit-learn class
# Author: Axel Marmoret
# 
# Adapted from: https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html, Author: Kushan <kushansharma1@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClassifierMixin

import as_seg.autosimilarity_computation as as_comp
import as_seg.data_manipulation as dm
from as_seg.baseline_segmenter.baseline_estimators import SegmenterScoringMixin

class CBMEstimator(BaseEstimator, SegmenterScoringMixin):
    """
    Scikit-learn class for the CBM algorithm. May be used for practicity, following the scikit-learn API.
    """
    def __init__(self, similarity_function="cosine", max_size=32, penalty_weight=1, penalty_func="modulo8", bands_number=7):
        """
        Constructor of the CBM estimator.

        Parameters
        ----------
        similarity_function : string, optional
            The similarity function to use for computing the autosimilarity.
            The default is "cosine".
        max_size : integer, optional
            The maximal size of segments.
            The default is 32.
        penalty_weight : float, optional
            The ponderation parameter for the penalty function.
            The default is 1.
        penalty_func : string, optional
            The type of penalty function to use.
            The default is "modulo8".
        bands_number : positive integer or None, optional
            The number of bands in the kernel.
            For the full kernel, bands_number must be set to None
            (or higher than the maximal size, but cumbersome)
            See [1] for details.
            The default is 7.
        """
        self.similarity_function = similarity_function
        self.max_size = max_size
        self.penalty_weight = penalty_weight
        self.penalty_func = penalty_func
        self.bands_number = bands_number
        self.algorithm_name = "CBM"

    def predict(self, barwise_features):
        """
        Perform Predictions

        If the estimator is not fitted, then raise NotFittedError
        """
        ssm_matrix = as_comp.switch_autosimilarity(barwise_features, similarity_type=self.similarity_function)
        segments = compute_cbm(ssm_matrix, max_size=self.max_size, penalty_weight=self.penalty_weight, 
                               penalty_func=self.penalty_func, bands_number = self.bands_number)[0]
        return segments
    
    def predict_in_seconds(self, barwise_features, bars):
        """
        Perform Predictions, and convert the segments from bars to seconds.

        If the estimator is not fitted, then raise NotFittedError
        """
        segments = self.predict(barwise_features)
        return dm.segments_from_bar_to_time(segments, bars)
    
    def predict_in_seconds_this_autosimilarity(self, ssm_matrix, bars):
        """
        Perform Predictions on a given autosimilarity matrix, and convert the segments from bars to seconds.

        If the estimator is not fitted, then raise NotFittedError
        """
        segments = compute_cbm(ssm_matrix, max_size=self.max_size, penalty_weight=self.penalty_weight, 
                               penalty_func=self.penalty_func, bands_number = self.bands_number)[0]
        return dm.segments_from_bar_to_time(segments, bars)