# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:09:10 2020

@author: amarmore

A file which contains all code regarding conversion of data, or information extraction from it
(typically getting the bars, converting segments in frontiers, sonifying segmentation or computing its Hit-Rate score, etc).
"""

import as_seg.model.errors as err

import numpy as np
import mir_eval
import scipy
import librosa
import math

## Best practice, but requires torch install, so disabled by default.
## Uncomment if torch is installed on your machine.
# import torch
# default_device = "cuda" if torch.cuda.is_available() else "cpu"
## By default, set the device to "cpu". NB: This will be way slower...
default_device = "cpu"

# %% Read and treat inputs
def get_bars_from_audio_madmom(song_path):
    """
    Returns the bars of a song, directly from its audio signal.
    Encapsulates the downbeat estimator from the madmom toolbox [1].
    
    NB1: Note that the estimation implicitely assumes 3 or 4 beats per bar.
    
    NB2: Note also that this function artificially adds bars at the end of the song, so that the estimation spans the entire song length.
    May/should be debated.

    Parameters
    ----------
    song_path : String
        Path to the desired song.

    Returns
    -------
    downbeats_times : list of tuples of float
        List of the estimated bars, as (start, end) times.
        
    References
    ----------
    [1] Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., & Widmer, G. (2016, October). 
    Madmom: A new python audio and music signal processing library. 
    In Proceedings of the 24th ACM international conference on Multimedia (pp. 1174-1178).

    """
    import madmom.features.downbeats as dbt
    act = dbt.RNNDownBeatProcessor()(song_path)
    proc = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
    song_beats = proc(act)
    downbeats_times = [song_beats[0][0]]
    
    for beat in song_beats[1:]: # The first beat is already added
        if beat[1] == 1: # If the beat is a downbeat
            downbeats_times.append(beat[0])
            
    song_length = act.shape[0]/100 # Total length of the song
    downbeats_times.append(song_length) # adding the last downbeat
    return frontiers_to_segments(downbeats_times)

def get_bars_from_audio_beat_this(song_path, checkpoint_path = "final0"):
    """
    Returns the bars of a song, directly from its audio signal.
    Encapsulates the downbeat estimator from the beat_this toolbox [2].

    Parameters
    ----------
    song_path : String
        Path to the desired song.
    checkpoint_path : String
        Path to the desired checkpoint.

    Returns
    -------
    downbeats_times : list of tuples of float
        List of the estimated bars, as (start, end) times.
    
    References
    ----------
    [2] Foscarin, F., Schlüter, J., & Widmer, G. (2024). 
    Beat this! Accurate beat tracking without DBN postprocessing. 
    In Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR).
    """
    from beat_this.inference import File2Beats
    file2beats = File2Beats(checkpoint_path=checkpoint_path, device=default_device, dbn=False)
    beats, downbeats = file2beats(song_path)
    downbeats.append(beats[-1]) # Not sure of this one, but let's see
    return frontiers_to_segments(downbeats)
    
def get_beats_from_audio_madmom(song_path):
    """
    Uses madmom to estimate the beats of a song, from its audio signal.
    """
    import madmom.features.beats as bt
    act = bt.TCNBeatProcessor()(song_path)
    proc = bt.BeatTrackingProcessor(fps=100)
    song_beats = proc(act)
    
    # beats_times = []    
    # if song_beats[0][1] != 1: # Adding a first downbeat at the start of the song
        # beats_times.append(0.1)
    # for beat in song_beats:
        # if beat[1] == 1: # If the beat is a downbeat
            # downbeats_times.append(beat[0])
            
    return frontiers_to_segments(list(song_beats))

def get_beats_from_audio_beat_this(song_path, checkpoint_path="final0"):
    """
    Returns the beats of a song, directly from its audio signal.
    Encapsulates the beat estimator from the beat_this toolbox [2].

    Parameters
    ----------
    song_path : String
        Path to the desired song.
    checkpoint_path : String
        Path to the desired checkpoint.

    Returns
    -------
    beats_times : list of tuples of float
        List of the estimated beats, as (start, end) times.
    
    References
    ----------
    [2] Foscarin, F., Schlüter, J., & Widmer, G. (2024). 
    Beat this! Accurate beat tracking without DBN postprocessing. 
    In Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR).
    """
    file2beats = File2Beats(checkpoint_path=checkpoint_path, device=default_device, dbn=False)
    beats, downbeats = file2beats(song_path)
    return frontiers_to_segments(beats)

def get_beats_from_audio_msaf(signal, sr, hop_length):
    """
    Uses MSAF to estimate the beats of a song, from its audio signal.
    """
    _, audio_percussive = librosa.effects.hpss(signal)
    
    # Compute beats
    _, beat_frames = librosa.beat.beat_track(y=audio_percussive, sr=sr, hop_length=hop_length)

    # To times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr,hop_length=hop_length)

    # TODO: Is this really necessary?
    if len(beat_times) > 0 and beat_times[0] == 0:
        beat_times = beat_times[1:]
        beat_frames = beat_frames[1:]
        
    return beat_times, beat_frames

# %% Read and treat inputs
def get_segmentation_from_txt(path, annotations_type, return_labels=False):
    """
    Reads the segmentation annotations, and returns it in a list of tuples (start, end, index as a number)
    This function has been developped for AIST and MIREX10 annotations, adapted for these types of annotations.
    It will not work with another set of annotation.

    Parameters
    ----------
    path : String
        The path to the annotation.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        
    Raises
    ------
    err.InvalidArgumentValueException
        If the type of annotations is neither AIST or MIREX10

    Returns
    -------
    segments : list of tuples (float, float, integer)
        The segmentation, formatted in a list of tuples, and with labels as numbers (easier to interpret computationnally).

    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    file_seg = open(path)
    segments = []
    unique_labels = []
    all_labels = []
    
    for part in file_seg.readlines():
        tupl = part.split("\t")
        all_labels.append(tupl[2])
        if tupl[2] not in unique_labels: # If label wasn't already found in this annotation
            idx = len(unique_labels)
            unique_labels.append(tupl[2])
        else: # If this label was found for another segment
            idx = unique_labels.index(tupl[2])
        if annotations_type == "AIST":
            segments.append(((int(tupl[0]) / 100), (int(tupl[1]) / 100), idx))
        elif annotations_type == "MIREX10":
            segments.append((round(float(tupl[0]), 3), round(float(tupl[1]), 3), idx))
        else:
            raise err.InvalidArgumentValueException("Annotations type not understood")
    if return_labels:
        return segments, all_labels
    return segments

def get_annotation_name_from_song(song_number, annotations_type):
    """
    Returns the name of the annotation of this song according to the desired annotation type
    
    Specificly designed for RWC Pop dataset, shouldn't be used otherwise.
    For now are available:
        - AIST annotations [1]
        - MIREX 10 annotations [2]
    
    Parameters
    ----------
    song_number : integer or string
        The number of the song (which is its name).
    annotations_type : string
        The desired type of annotation.

    Raises
    ------
    InvalidArgumentValueException
        If the annotatipn type is not implemented.

    Returns
    -------
    string
        The name of the file containing the annotation.
        
    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    if annotations_type == "MIREX10":
        return "RM-P{:03d}.BLOCKS.lab".format(int(song_number))
    elif annotations_type == "AIST":
        return "RM-P{:03d}.CHORUS.TXT".format(int(song_number))
    else:
        raise err.InvalidArgumentValueException("Annotations type not understood")

# %% Segments -> frontiers and frontiers -> segments
def frontiers_to_segments(frontiers_in):
    """
    Computes a list of segments starting from the frontiers between them.

    Parameters
    ----------
    frontiers : list of floats
        The list of frontiers.

    Returns
    -------
    to_return : list of tuples of floats
        The segments, as tuples (start, end).

    """
    to_return = []
    frontiers = list(set(frontiers_in))
    frontiers.sort()
    # if remove_zeroes:
    #     if 0 in frontiers:
    #         frontiers.remove(0)
    for idx in range(len(frontiers) - 1):
        if frontiers[idx] != frontiers[idx + 1]:
            to_return.append((frontiers[idx], frontiers[idx + 1]))
    return np.array(to_return)

def segments_to_frontiers(segments):
    """
    Computes a list of frontiers from the segments.

    Parameters
    ----------
    segments : list of tuples of floats
        The segments, as tuples.

    Returns
    -------
    list
        Frontiers between segments.

    """
    to_return = [i[0] for i in segments]
    to_return.append(segments[-1][1]) # Adding the last frontier. Poteaux et intervalles.
    return np.array(to_return)

# %% Conversion of data (time/frame/bar)
def frontiers_from_time_to_frame_idx(seq, hop_length_seconds):
    """
    Converts a sequence of frontiers in time to their values in frame indexes.

    Parameters
    ----------
    seq : list of float/times
        The list of times to convert.
    hop_length_seconds : float
        hop_length (time between two consecutive frames), in seconds.

    Returns
    -------
    list of integers
        The sequence, as a list, in frame indexes.
    """
    
    return [int(round(frontier/hop_length_seconds)) for frontier in seq]

def segments_from_time_to_frame_idx(segments, hop_length_seconds):
    """
    Converts a sequence of segments (start, end) in time to their values in frame indexes.

    Parameters
    ----------
    segements : list of tuple
        The list of segments, as tuple (start, end), to convert.
    hop_length_seconds : float
        hop_length (time between two consecutive frames), in seconds.

    Returns
    -------
    list of integers
        The sequence, as a list, in frame indexes.
    """
    to_return = []
    for segment in segments:
        bar_in_frames = [int(round(segment[0]/hop_length_seconds)), int(round(segment[1]/hop_length_seconds))]
        if bar_in_frames[0] != bar_in_frames[1]: # Remove empty segments
            to_return.append(bar_in_frames)
    return to_return

## Bar -> time
def frontiers_from_bar_to_time(seq, bars):
    """
    Converts frontiers from bar indexes to absolute times.
    Convention: Boundary `i` is the START of bar `i`.
    """
    to_return = []
    for frontier in seq:
        if frontier < len(bars):
            t = bars[frontier][0]
        elif frontier == len(bars):
            t = bars[-1][1]
        else:
            raise ValueError("Frontier {} is out of bounds for {} bars.".format(frontier, len(bars)))
        if t not in to_return:
            to_return.append(t)
    return to_return

def segments_from_bar_to_time(segments, bars):
    """
    Converts segments from bar indexes to time.

    Parameters
    ----------
    segments : list of tuple of integers
        The indexes of the bars defining the segments (start, end).
    bars : list of tuple of float
        Bars, as tuples (start, end), in time.

    Returns
    -------
    numpy array
        Segments, in time.

    """
    frontiers = segments_to_frontiers(segments)
    frontiers_in_time = frontiers_from_bar_to_time(frontiers, bars)
    to_return = frontiers_to_segments(frontiers_in_time)
    return np.array(to_return)

## Time -> bar index
def frontiers_from_time_to_bar(seq, bars):
    """
    Converts the frontiers in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.

    Parameters
    ----------
    seq : list of float
        The list of frontiers, in time.
    bars : list of tuple of floats
        The bars, as (start time, end time) tuples.

    Returns
    -------
    seq_barwise : list of integers
        List of times converted in bar indexes.

    """
    seq_barwise = []
    for frontier in seq:
        # Edge cases first:
        if frontier < bars[0][0]:
            seq_barwise.append(0)
        elif frontier >= bars[-1][1]:
            seq_barwise.append(len(bars)) # Careful, this may lead to an out of bound index error
        # General case:
        else:
            for idx, bar in enumerate(bars):
                if frontier >= bar[0] and frontier < bar[1]: # The frontier is in the bar
                    if bar[1] - frontier < frontier - bar[0]: # The frontier is closer to the end of the bar
                        seq_barwise.append(idx+1) # Careful, of the edge case of the last bar.
                    else:
                        seq_barwise.append(idx)
                    break
    seq_barwise = np.array(sorted(list(set(seq_barwise)))) # Removing duplicates and sorting
    return seq_barwise

def segments_from_time_to_bar(seq, bars):
    """
    Converts the segments in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.
    """
    frontiers = segments_to_frontiers(seq)
    frontiers_in_bar = frontiers_from_time_to_bar(frontiers, bars)
    to_return = frontiers_to_segments(frontiers_in_bar)
    return np.array(to_return)

## time -> time, but bar aligned
def align_frontiers_on_bars(seq, bars):
    """
    Converts the frontiers in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.

    Parameters
    ----------
    seq : list of float
        The list of frontiers, in time.
    bars : list of tuple of floats
        The bars, as (start time, end time) tuples.

    Returns
    -------
    seq_barwise : list of integers
        List of times converted in bar indexes.

    """
    seq_barwise = []
    for frontier in seq:
        # Edge cases first:
        if frontier < bars[0][0]:
            seq_barwise.append(bars[0][0])
        elif frontier >= bars[-1][1]:
            seq_barwise.append(bars[-1][1])
        # General case:
        else:
            for idx, bar in enumerate(bars):
                if frontier >= bar[0] and frontier < bar[1]:
                    if bar[1] - frontier < frontier - bar[0]:
                        seq_barwise.append(bar[1])
                    else:
                        seq_barwise.append(bar[0])
                    break
    seq_barwise = np.array(sorted(list(set(seq_barwise)))) # Removing duplicates and sorting
    return seq_barwise

def align_segments_on_bars(seq, bars):
    """
    Converts the segments in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.
    """
    frontiers = segments_to_frontiers(seq)
    frontiers_in_bar = align_frontiers_on_bars(frontiers, bars)
    to_return = frontiers_to_segments(frontiers_in_bar)
    return np.array(to_return)

# %% Audio reconstruction
def get_median_hop(bars, subdivision = 96, sampling_rate = 44100):
    """
    Returns the median hop length in the song, used for audio reconstruction.
    The rationale is that all bars are sampled with 'subdivision' number of frames, 
    but they can be of different lengths in absolute time.
    Hence, the time gap between two consecutive frames (the hop length) can differ between bars.
    For reconstruction, we use the median hop length among all bars.

    Parameters
    ----------
    bars : list of tuples of float
        The bars, as (start time, end time) tuples.
    subdivision : integer, optional
        The number of subdivision of the bar to be contained in each slice of the tensor.
        The default is 96.
    sampling_rate : integer, optional
        The sampling rate of the signal, in Hz.
        The default is 44100.

    Returns
    -------
    integer
        The median hop length in these bars.

    """
    hops = []
    for bar_idx in range(1, len(bars)):
        len_sig = bars[bar_idx][1] - bars[bar_idx][0]
        hop = int(len_sig/subdivision * sampling_rate)
        hops.append(hop)
    return int(np.median(hops)) # Generally used for audio reconstruction

# %% Sonification of the segmentation
def sonify_frontiers_path(audio_file_path, frontiers_in_seconds, output_path):
    """
    Takes the path of the song and frontiers, and write a song with the frontiers sonified ("bip" in the song).
    Function inspired from MSAF.

    Parameters
    ----------
    audio_file_path: String
        The path to the song, (as signal).
    frontiers_in_seconds: list of floats
        The frontiers, in time/seconds.
    output_path: String
        The path where to write the song with sonified frontiers.

    Returns
    -------
    Nothing, but writes a song at output_path

    """
    the_signal, sampling_rate = librosa.load(audio_file_path, sr=None)
    sonify_frontiers_song(the_signal, sampling_rate, frontiers_in_seconds, output_path)

def sonify_frontiers_song(song_signal, sampling_rate, frontiers_in_seconds, output_path):
    """
    Takes a song as a signal, and add the frontiers to this signal.
    It then writes it as a file.
    Function inspired from MSAF.

    Parameters
    ----------
    song_signal : numpy array
        The song as a signal.
    sampling_rate : integer
        The sampling rate of the signal, in Hz.
    frontiers_in_seconds: list of floats
        The frontiers, in time/seconds.
    output_path: String
        The path where to write the song with sonified frontiers.

    Returns
    -------
    Nothing, but writes a song at the output_path.

    """
    frontiers_signal = mir_eval.sonify.clicks(frontiers_in_seconds, sampling_rate)
    if song_signal.ndim == 2:
        song_signal = song_signal[:,0]
    signal_with_frontiers = np.zeros(max(len(song_signal), len(frontiers_signal)))
    signal_with_frontiers[:len(song_signal)] = song_signal
    signal_with_frontiers[:len(frontiers_signal)] += frontiers_signal
    
    scipy.io.wavfile.write(output_path, sampling_rate, signal_with_frontiers)
    
# %% Score calculation encapsulation
def compute_score_from_frontiers_in_bar(reference, frontiers_in_bar, bars, window_length = 0.5, trim = False):
    """
    Computes precision, recall and f measure from estimated frontiers (in bar indexes) and the reference (in seconds).
    Scores are computed from the mir_eval toolbox.

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    frontiers : list of integers
        The frontiers between segments, in bar indexes.
    bars : list of tuples
        The bars of the song.
    window_length : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.

    Returns
    -------
    precision: float \\in [0,1]
        Precision of these frontiers,
        ie the proportion of accurately found frontiers among all found frontiers.
    recall: float \\in [0,1]
        Recall of these frontiers,
        ie the proportion of accurately found frontiers among all accurate frontiers.
    f_measure: float \\in [0,1]
        F measure of these frontiers,
        ie the geometric mean of both precedent scores.
        
    """
    try:
        np.array(bars).shape[1]
    except:
        raise err.OutdatedBehaviorException("Bars is still a list of downbeats, which is an old beavior, and shouldn't happen anymore. To track and to fix.")
    frontiers_in_time = frontiers_from_bar_to_time(frontiers_in_bar, bars)
    return compute_score_of_segmentation(reference, frontiers_to_segments(frontiers_in_time), window_length = window_length, trim = trim)

def compute_score_of_segmentation(reference, segments_in_time, window_length = 0.5, trim = False):
    """
    Computes precision, recall and f measure from estimated segments and the reference, both in seconds.    
    Scores are computed from the mir_eval toolbox.

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The segments, in seconds, as tuples (start, end).
    window_length : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.
    trim : boolean, optional
        Whether (True) or not (False) the first and last boundaries should be "trimmed", 
        i.e. discarded in the computation of accurate boundaries.
        Default is False, meaning that they are kept by default.
        This may artifically increase the metrics, but is the standard in the benchmarks.

    Returns
    -------
    precision: float \\in [0,1]
        Precision of these frontiers,
        ie the proportion of accurately found frontiers among all found frontiers.
    recall: float \\in [0,1]
        Recall of these frontiers,
        ie the proportion of accurately found frontiers among all accurate frontiers.
    f_measure: float \\in [0,1]
        F measure of these frontiers,
        ie the geometric mean of both precedent scores.

    """
    # # adjust_intervals is actually adding start and ending intevals, leading to somehow inconsistent results with the trimming I want.
    # ref_intervals, useless = mir_eval.util.adjust_intervals(reference,t_min=0)
    # est_intervals, useless = mir_eval.util.adjust_intervals(np.array(segments_in_time), t_min=0) #, t_max=ref_intervals[-1, 1])
    ref_intervals = np.array(reference)
    est_intervals = np.array(segments_in_time)
    try:
        return mir_eval.segment.detection(ref_intervals, est_intervals, window = window_length, trim = trim)
    except ValueError:
        cleaned_intervals = []
        #print("A segment is (probably) composed of the same start and end. Can happen with time -> bar -> time conversion, but should'nt happen for data originally segmented in bars.")
        for idx in range(len(est_intervals)):
            if est_intervals[idx][0] != est_intervals[idx][1]:
                cleaned_intervals.append(est_intervals[idx])
        return mir_eval.segment.detection(ref_intervals, np.array(cleaned_intervals), window = window_length, trim = trim)


def compute_score_of_segmentation_barscale(reference, segments_in_time, bars, window_length = 0, trim = False):
    """
    Computes precision, recall and f measure from estimated segments and the reference, both aligned on bars.
    Reference and segments are converted from time to bar indexes before scoring,
    so window_length is expressed in bars rather than seconds.
    Scores are computed from the mir_eval toolbox.

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The estimated segments, in seconds, as tuples (start, end).
    bars : list of tuples of float
        The bars of the song, as (start time, end time) tuples.
    window_length : integer, optional
        The window size for the score, expressed in bars.
        0 means exact bar match, 1 means one-bar tolerance.
        The default is 0.
    trim : boolean, optional
        Whether (True) or not (False) the first and last boundaries should be "trimmed".
        Default is False.

    Returns
    -------
    precision: float \\in [0,1]
        Precision of these frontiers.
    recall: float \\in [0,1]
        Recall of these frontiers.
    f_measure: float \\in [0,1]
        F measure of these frontiers.

    """
    ref_in_bars = np.array(segments_from_time_to_bar(reference, bars))
    est_in_bars = np.array(segments_from_time_to_bar(segments_in_time, bars)) # Probably useless, as it should already be in bars, but just in case.
    return compute_score_of_segmentation(ref_in_bars, est_in_bars, window_length=window_length, trim=trim)


def compute_multiple_scores_of_segmentation(references_list, segments_in_time, window_length=0.5, trim=False, reduce_method="mean"):
    """
    Computes scores for multiple references (e.g. from multiple annotators/levels) against predictions.
    If `segments_in_time` is a list, it evaluates the i-th prediction against the i-th reference.
    
    Parameters
    ----------
    references_list : list of np.ndarray
        List of ground-truth segments.
    segments_in_time : np.ndarray or list of np.ndarray
        Predicted segments.
    window_length : float, optional
        Tolerance window for boundaries comparison. The default is 0.5.
    trim : bool, optional
        Whether to trim the first and last boundaries before scoring using mir_eval. Default is False.
    reduce_method : str, optional
        How to reduce the scores ("mean" or "max"). "max" selects the score tuple with the highest f-measure. The default is "mean".

    Returns
    -------
    tuple
        (precision, recall, f-measure)
    """
    scores = []
    for i in range(len(references_list)):
        pred = segments_in_time[i] if isinstance(segments_in_time, list) else segments_in_time
        score = compute_score_of_segmentation(references_list[i], pred, window_length=window_length, trim=trim)
        scores.append(score)
    
    if reduce_method == "mean":
        return tuple(np.mean(scores, axis=0))
    elif reduce_method == "max":
        best_idx = np.argmax([s[2] for s in scores])
        return scores[best_idx]
    else:
        raise ValueError(f"Unknown reduce_method: {reduce_method}. Expected 'mean' or 'max'.")

def compute_median_deviation_of_segmentation(reference, segments_in_time):
    """
    TODO

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The segments, in seconds, as tuples (start, end).

    Returns
    -------
    TODO
    r_to_e then e_to_r

    """
    # ref_intervals, useless = mir_eval.util.adjust_intervals(reference,t_min=0)
    # est_intervals, useless = mir_eval.util.adjust_intervals(np.array(segments_in_time), t_min=0) #, t_max=ref_intervals[-1, 1])
    try:
        return mir_eval.segment.deviation(ref_intervals,est_intervals)
    except ValueError:
        cleaned_intervals = []
        for idx in range(len(est_intervals)):
            if est_intervals[idx][0] != est_intervals[idx][1]:
                cleaned_intervals.append(est_intervals[idx])
        return mir_eval.segment.deviation(ref_intervals,est_intervals)

def compute_rates_of_segmentation(reference, segments_in_time, window_length = 0.5):
    """
    Computes True Positives, False Positives and False Negatives from estimated segments and the reference, both in seconds.    
    Scores are computed from the mir_eval toolbox. (In fact, the code is extracted from mir_eval.segment.detection)

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The segments, in seconds, as tuples (start, end).
    window_length : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.

    Returns
    -------
    True Positives: Integer
        The number of True Positives, 
        ie the number of accurately found frontiers.
    False Positives: Integer
        The number of False Positives,
        ie the number of wrongly found frontiers (estimated frontiers which are incorrect).
    False Negative : Integer
        The number of False Negatives,
        ie the number of frontiers undetected (accurate frontiers which are not found in teh estimation).

    """  
    # reference_intervals, _ = mir_eval.util.adjust_intervals(reference,t_min=0)
    # estimated_intervals, _ = mir_eval.util.adjust_intervals(segments_in_time, t_min=0) #, t_max=reference_intervals[-1, 1])
    
    mir_eval.segment.validate_boundary(reference_intervals, estimated_intervals, False)

    # Convert intervals to boundaries
    reference_boundaries = mir_eval.util.intervals_to_boundaries(reference_intervals)
    estimated_boundaries = mir_eval.util.intervals_to_boundaries(estimated_intervals)

    # If we have no boundaries, we get no score.
    if len(reference_boundaries) == 0 or len(estimated_boundaries) == 0:
        return 0, 0, 0

    tp = len(mir_eval.util.match_events(reference_boundaries,estimated_boundaries,window_length))
    fp = len(estimated_boundaries) - tp
    fn = len(reference_boundaries) - tp
    
    return tp, fp, fn
