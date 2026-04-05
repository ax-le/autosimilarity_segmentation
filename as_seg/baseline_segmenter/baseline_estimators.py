import numpy as np
from sklearn.base import BaseEstimator
import as_seg.data_manipulation as dm

class SegmenterScoringMixin:
    """
    Mixin class that provides a standard `score` method for all structural segmentation estimators.
    """
    def score(self, predictions, annotations, trim=False, my_trim_flag=None, len_signal=None, labels=None, reduce_method="mean"):
        """
        Compute the score of the predictions.
        
        Parameters
        ----------  
        predictions : np.ndarray
            Predicted segments as (start, end) pairs in time (seconds).
        annotations : np.ndarray or list of np.ndarray
            Ground-truth annotation segments (could be a list if multiple annotators).
        trim : bool, optional
            mir_eval trim flag — discard first/last boundaries. Default False.
        my_trim_flag : bool or None, optional
            If not None, apply silent-segment handling via apply_my_trim.
            True  -> remove silent segments from annotations and predictions.
            False -> add synthetic silent segments to predictions.
            None  -> do nothing (default).
        len_signal : float or None
            Duration of the audio signal in seconds (required when my_trim_flag is not None).
        labels : list of str or None
            Annotation labels (required when my_trim_flag is True).
        reduce_method : str, optional
            How to reduce multiple annotations to a single score. "mean" or "max".
        """
        if my_trim_flag is not None:
            # Import dynamically to avoid circular dependencies
            import as_seg.model.trimming_utils as tmu
            annotations, predictions = tmu.apply_my_trim(
                annotations, predictions, labels, len_signal, my_trim_flag
            )

        if isinstance(annotations, list): # Salami and multiple annotations
            close_tolerance = dm.compute_multiple_scores_of_segmentation(annotations, predictions, window_length=0.5, trim=trim, reduce_method=reduce_method)
            large_tolerance = dm.compute_multiple_scores_of_segmentation(annotations, predictions, window_length=3, trim=trim, reduce_method=reduce_method)
            return close_tolerance, large_tolerance

        close_tolerance = dm.compute_score_of_segmentation(annotations, predictions, window_length=0.5, trim=trim)
        large_tolerance = dm.compute_score_of_segmentation(annotations, predictions, window_length=3, trim=trim)
        return close_tolerance, large_tolerance

class FooteEstimator(BaseEstimator, SegmenterScoringMixin):
    """
    Foote novelty-based segmentation algorithm exposed as a scikit-learn style estimator.
    """
    def __init__(self, similarity_function="cosine", M_gaussian=16, L_peaks=16, pre_filter=0, post_filter=0):
        self.similarity_function = similarity_function
        self.M_gaussian = M_gaussian
        self.L_peaks = L_peaks
        self.pre_filter = pre_filter
        self.post_filter = post_filter

    def predict(self, barwise_features):
        """
        Predict structural segments using the Foote algorithm from barwise TF features.
        
        Parameters
        ----------
        barwise_features : np.ndarray
            Barwise features (e.g. embeddings or spectrograms).
            
        Returns
        -------
        list
            Predicted segments in bars.
        """
        import as_seg.autosimilarity_computation as as_comp
        from as_seg.baseline_segmenter.foote.foote_api import process_msaf_foote
        
        ssm_matrix = as_comp.switch_autosimilarity(barwise_features, similarity_type=self.similarity_function)
        
        barwise_foote_bndr = process_msaf_foote(
            input_ssm=ssm_matrix, 
            M_gaussian=self.M_gaussian, 
            L_peaks=self.L_peaks, 
            pre_filter=self.pre_filter, 
            post_filter=self.post_filter
        )[0]
        
        return dm.frontiers_to_segments(list(barwise_foote_bndr))

    def predict_in_seconds(self, barwise_features, bars):
        """
        Predict structural segments using the Foote algorithm from barwise TF features.
        
        Parameters
        ----------
        barwise_features : np.ndarray
            Barwise features (e.g. embeddings or spectrograms).
        bars : list of tuples
            Bar start and end times in seconds.
            
        Returns
        -------
        np.ndarray
            Predicted segments in time (seconds).
        """
        barwise_segments = self.predict(barwise_features)
        return dm.segments_from_bar_to_time(barwise_segments, bars)

    def predict_in_seconds_this_autosimilarity(self, ssm_matrix, bars):
        """
        Perform Predictions on a given autosimilarity matrix, and convert the segments from bars to seconds.

        If the estimator is not fitted, then raise NotFittedError
        """
        from as_seg.baseline_segmenter.foote.foote_api import process_msaf_foote

        barwise_foote_bndr = process_msaf_foote(
            input_ssm=ssm_matrix, 
            M_gaussian=self.M_gaussian, 
            L_peaks=self.L_peaks, 
            pre_filter=self.pre_filter, 
            post_filter=self.post_filter
        )[0]
        
        segments = dm.frontiers_to_segments(list(barwise_foote_bndr))
        return dm.segments_from_bar_to_time(segments, bars)

class LSDEstimator(BaseEstimator, SegmenterScoringMixin):
    """
    Laplacian Spectral Decomposition (LSD) segmentation algorithm exposed as a scikit-learn style estimator.
    """
    def __init__(self, scluster_k=None, evec_smooth=None, rec_smooth=None, rec_width=None):
        self.scluster_k = scluster_k
        self.evec_smooth = evec_smooth
        self.rec_smooth = rec_smooth
        self.rec_width = rec_width

    def predict(self, barwise_tf_matrix):
        """
        Predict structural segments using the LSD algorithm.
        
        Parameters
        ----------
        barwise_tf_matrix : np.ndarray
            Barwise time-frequency matrix (embeddings).
            
        Returns
        -------
        list
            Predicted segments in bars.
        """
        from as_seg.baseline_segmenter.lsd.lsd_api import process_msaf_lsd
        
        est_lsd = process_msaf_lsd(
            barwise_tf_matrix,
            scluster_k=self.scluster_k,
            evec_smooth=self.evec_smooth,
            rec_smooth=self.rec_smooth,
            rec_width=self.rec_width,
        )[0]
        
        return dm.frontiers_to_segments(list(est_lsd))

    def predict_in_seconds(self, barwise_tf_matrix, bars):
        """
        Predict structural segments using the LSD algorithm.
        
        Parameters
        ----------
        barwise_tf_matrix : np.ndarray
            Barwise time-frequency matrix (embeddings).
        bars : list of tuples
            Bar start and end times in seconds.
            
        Returns
        -------
        np.ndarray
            Predicted segments in time (seconds).
        """
        barwise_segments = self.predict(barwise_tf_matrix)
        return dm.segments_from_bar_to_time(barwise_segments, bars)
