# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:34:39 2020

@author: amarmore

A module containing some high-level scripts for decomposition and/or segmentation.
Not meant to be shared but rather to work.
"""

import librosa.core
import librosa.feature
import librosa
import os
import numpy as np

import as_seg.data_manipulation as dm
import as_seg.barwise_input as bi
import as_seg.model.signal_to_spectrogram as signal_to_spectrogram
import as_seg.model.errors as err
import as_seg.scripts.default_path as paths


# %% SALAMI test dataset
def get_salami_test_indexes():
    list_songs_in_test = open(f"{paths.path_data_persisted_salami}/test_set_salami_gs.txt")
    test_dataset = []
    for part in list_songs_in_test.readlines():
        line_broken = part.split("\n")
        test_dataset.append(int(line_broken[0]))
    return test_dataset

def is_in_salami_train(key, salami_test_dataset = None):
    if salami_test_dataset is None: # May be passed as argument to avoid loading the file at each test.
        salami_test_dataset = get_salami_test_indexes()
    return not key in salami_test_dataset

def is_in_salami_test(key, salami_test_dataset = None):
    if salami_test_dataset is None: # May be passed as argument to avoid loading the file at each test.
        salami_test_dataset = get_salami_test_indexes()
    return key in salami_test_dataset

# %% Loading (or saving) multiple data
def load_spec_annot_song_RWC(song_number, feature, hop_length = 32):
    """
    Load the spectrogram, the bar segmentation and the annotation for a given song in RWC Pop.
    For this function to work, paths should be updated to correct ones.

    Parameters
    ----------
    song_number : integer
        Index of the song in the RWC Pop dataset (1 to 100).
    feature : string
        The feature description to compute the spectrogram with.
    hop_length : int, optional
        Hop length in the computation of the spetrogram.
        The default is 32.

    Returns
    -------
    spectrogram : numpy array
        The spectorgram of this song, searched in memory.
    bars : list of tuple of floats
        The bars for this song, searched in memory.
    references_segments : np.array
        Segments of reference (annotations) for this song.
    """
    spectrogram = load_spectrogram("rwcpop", song_number, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_bars("rwcpop", song_number)
    annotations_mirex = f"{paths.path_annotation_rwc}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

def load_or_save_spec_annot_song_RWC(song_number, feature, hop_length = 32):
    """
    Load the spectrogram, the bar segmentation and the annotation for a given song in RWC Pop.
    For this function to work, paths should be updated to correct ones,
    and outputs will be stored if not computed (spectrograms and bars typically).

    Parameters
    ----------
    song_number : integer
        Index of the song in the RWC Pop dataset (1 to 100).
    feature : string
        The feature description to compute the spectrogram with.
    hop_length : int, optional
        Hop length in the computation of the spetrogram.
        The default is 32.

    Returns
    -------
    spectrogram : numpy array
        The spectorgram of this song, either computed or found in memory.
    bars : list of tuple of floats
        The bars for this song, either computed or found in memory.
    references_segments : np.array
        Segments of reference (annotations) for this song.
    """
    spectrogram = load_or_save_spectrogram("rwcpop", f"{paths.path_entire_rwc}/{song_number}.wav", feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_or_save_bars("rwcpop", f"{paths.path_entire_rwc}/{song_number}.wav")
    annotations_mirex = f"{paths.path_annotation_rwc}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

def load_bar_annot_song_RWC(song_number):
    """
    Similar to load_spec_annot_song_RWC(), but without loading the soectrogram 
    (only bars and annotations).
    
    Generally used when spectrogram is not desired but has a huge memory cost (or may not be known).
    """
    bars = load_bars("rwcpop", song_number)
    annotations_mirex = f"{paths.path_annotation_rwcpop}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return bars, references_segments

def load_or_save_bar_annot_song_RWC(song_number):
    """
    Similar to load_or_save_spec_annot_song_RWC(), but without the spectrogram 
    (only bars and annotations).
    
    Generally used when spectrogram is not desired but has a huge computation or memory cost.
    """
    bars = load_or_save_bars("rwcpop", f"{paths.path_entire_rwc}/{song_number}.wav")
    annotations_mirex = f"{paths.path_annotation_rwc}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return bars, references_segments

def load_beat_annot_song_RWC(song_number):
    """
    Loads beats and annotations for this song.
    """
    beats = load_beats("rwcpop", song_number)
    annotations_mirex = f"{paths.path_annotation_rwcpop}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return beats, references_segments

def load_or_save_spectrogram_and_bars(dataset, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Loads the spectrogram and the bars for this song, which were persisted after a first computation, or compute them if they weren't found.

    Parameters
    ----------
    dataset : string
        Name of the dataset. Only "rwcpop" and "salami" for now.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.
    n_fft and n_mfcc : integers, optional
        Both arguments are used respectively for the stft and for the mfcc computation, and are used to 

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    bars = load_or_save_bars(dataset, song_path)
    spectrogram = load_or_save_spectrogram(dataset, song_path, feature, hop_length, fmin = fmin, n_fft = n_fft, n_mfcc = n_mfcc)
    return bars, spectrogram

def load_or_save_bars_and_barwise_tf(dataset, song_path, feature = "log_mel_grill", subdivision = 96, hop_length=32):
    bars = load_or_save_bars(dataset, song_path)
    barwise_tf = load_or_save_barwise_tf(dataset, song_path, bars, feature, subdivision, hop_length)
    return bars, barwise_tf

def load_or_save_beats_and_beatwise_tf(dataset, song_path, feature = "log_mel_grill", subdivision_beat = 24, hop_length=32):
    beats = load_or_save_beats(dataset, song_path)
    beatwise_tf = load_or_save_beatwise_tf(dataset, song_path, beats, feature, subdivision_beat, hop_length)
    return beats, beatwise_tf

def load_spec_annot_cometogether(feature, hop_length = 32):
    """
    Load the spectrogram, the bar segmentation and the annotation for Come Together, from the Beatles.
    For this function to work, paths should be updated to correct ones,
    and outputs will be stored if not computed (spectrograms and bars typically).

    Parameters
    ----------
    feature : string
        The feature description to compute the spectrogram with.
    hop_length : int, optional
        Hop length in the computation of the spetrogram.
        The default is 32.

    Returns
    -------
    spectrogram : numpy array
        The spectorgram of this song, either computed or found in memory.
    bars : list of tuple of floats
        The bars for this song, either computed or found in memory.
    references_segments : np.array
        Segments of reference (annotations) for this song.
    """
    spectrogram = load_spectrogram("cometogether", "come_together", feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_bars("cometogether", f"{paths.come_together}.wav")
    annotation_path = f"{paths.come_together}.lab"
    annotations = dm.get_segmentation_from_txt(annotation_path, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

def load_or_save_spec_annot_cometogether(feature, hop_length = 32):
    """
    Load the spectrogram, the bar segmentation and the annotation for Come Together, from the Beatles.
    For this function to work, paths should be updated to correct ones,
    and outputs will be stored if not computed (spectrograms and bars typically).

    Parameters
    ----------
    feature : string
        The feature description to compute the spectrogram with.
    hop_length : int, optional
        Hop length in the computation of the spetrogram.
        The default is 32.

    Returns
    -------
    spectrogram : numpy array
        The spectorgram of this song, either computed or found in memory.
    bars : list of tuple of floats
        The bars for this song, either computed or found in memory.
    references_segments : np.array
        Segments of reference (annotations) for this song.
    """
    spectrogram = load_or_save_spectrogram("cometogether", f"{paths.come_together}.wav", feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_or_save_bars("cometogether", f"{paths.come_together}.wav")
    annotation_path = f"{paths.come_together}.lab"
    annotations = dm.get_segmentation_from_txt(annotation_path, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

# %% Loading (or saving) spectrograms
def load_spectrogram(dataset, song_name, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Lloads the spectrogram for this song (supposing it was already computed).

    Parameters
    ----------
    dataset : string
        Name of the dataset (should be either "rwcpop" or "salami")
    song_name : string
        The name of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------=
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    name_spectrogram = format_name_spectrogram_params(song_name, feature, hop_length, fmin, n_fft, n_mfcc)
    return np.load(f"{paths.path_data_persisted}/{dataset}/spectrograms/{name_spectrogram}.npy")

def load_or_save_spectrogram(dataset, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Computes the spectrogram for this song, or load it if it were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------=
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        return load_spectrogram(dataset, song_name, feature, hop_length, fmin, n_fft, n_mfcc)
    except FileNotFoundError:
        if "log_mel_grill" in feature:
            mel = load_or_save_spectrogram(dataset, song_path, "mel_grill", hop_length, fmin = fmin, n_fft = n_fft, n_mfcc = n_mfcc)
            return signal_to_spectrogram.get_log_mel_from_mel(mel, feature)
        elif feature == "mel" or feature == "log_mel" or feature == "logmel":
            raise err.InvalidArgumentValueException(f"Invalid mel parameter ({feature}), are't you looking for mel_grill?")
        else:
            the_signal, _ = librosa.load(song_path, sr=44100)
            if "stft" in feature:
                if "nfft" not in feature: 
                    spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft)
                else:              
                    n_fft_arg = int(feature.split("nfft")[1])
                    spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft_arg)
            elif "mfcc" in feature:
                if "nmfcc" not in feature:
                    spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)   
                else:
                    n_mfcc_arg = int(feature.split("nmfcc")[1])
                    spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
            elif feature == "pcp_tonnetz":
                # If chromas are already computed, try to load them instead of recomputing them.
                chromas = load_or_save_spectrogram(dataset, song_path, "pcp", hop_length, fmin = fmin)
                spectrogram = librosa.feature.tonnetz(y=None, sr = None, chroma = chromas)
            elif feature == "pcp":
                # If it wasn't pcp_tonnetz, compute the spectrogram, and then save it.
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
            else:
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, feature, hop_length)
        
            name_spectrogram = format_name_spectrogram_params(song_name, feature, hop_length, fmin, n_fft, n_mfcc)
            np.save(f"{paths.path_data_persisted}/{dataset}/spectrograms/{name_spectrogram}", spectrogram)
            return spectrogram

def format_name_spectrogram_params(song_name, feature, hop_length, fmin, n_fft, n_mfcc):
    if "stft" in feature:   
        if "nfft" not in feature:
            return f"{song_name}_{feature}-nfft{n_fft}_stereo_{hop_length}"
        else:
            return f"{song_name}_{feature}_stereo_{hop_length}"
    elif "mfcc" in feature:
        if "nmfcc" not in feature:
            return f"{song_name}_{feature}-nmfcc{n_mfcc}_stereo_{hop_length}"
        else:
            return f"{song_name}_{feature}_stereo_{hop_length}"
    elif feature == "pcp":
        return f"{song_name}_{feature}_stereo_{hop_length}_{fmin}"
    elif feature == "mel" or feature == "log_mel":
        raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
    else:
        return f"{song_name}_{feature}_stereo_{hop_length}"
    
# %% Loading (or saving) bars and beats
def load_bars(dataset, song_name):
    """
    Loads the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    dataset : string
        Name of the dataset. For now are only handled "salami" and "rwcpop".
    song_name : string
        Name of the song (identifier of the bars to load).

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    return np.load(f"{paths.path_data_persisted}/{dataset}/bars/{song_name}.npy")

def load_or_save_bars(dataset, song_path):
    """
    Computes the bars for this song, or load them if they were already computed.

    Parameters
    ----------
    dataset : string
        Name of the dataset. Only "rwcpop" and "salami" for now.
    song_path : string
        The path of the signal of the song.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        bars = load_bars(dataset, song_name)
    except:
        bars = dm.get_bars_from_audio(song_path)
        np.save(f"{paths.path_data_persisted}/{dataset}/bars/{song_name}", bars)
    return bars

def load_beats(dataset, song_name):
    """
    Loads the beats for this song, which were persisted after a first computation.

    Parameters
    ----------
    dataset : string
        Name of the dataset. For now are only handled "salami" and "rwcpop".
    song_name : string
        Name of the song (identifier of the bars to load).

    Returns
    -------
    beats : list of tuple of floats
        The persisted beats for this song.
    """
    return np.load(f"{paths.path_data_persisted}/{dataset}/beats/{song_name}.npy")

def load_or_save_beats(dataset, song_path):
    """
    Computes the beats for this song, or load them if they were already computed.

    Parameters
    ----------
    dataset : string
        Name of the dataset. Only "rwcpop" and "salami" for now.
    song_path : string
        The path of the signal of the song.

    Returns
    -------
    beats : list of tuple of floats
        The persisted beats for this song.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        beats = load_beats(dataset, song_name)
    except:
        beats = dm.get_beats_from_audio_madmom(song_path)
        np.save(f"{paths.path_data_persisted}/{dataset}/beats/{song_name}", beats)
    return beats

# %% load (or save) TF matrices
def get_name_tf_matrices(dataset, song_name, feature, subdivision):
    return f"{dataset}_{song_name}_{feature}_subdiv{subdivision}"

def load_barwise_tf(dataset, song_name, feature = "log_mel_grill", subdivision = 96):
    """
    Load the barwiseTF matrix (already pre-computed) for this song.
    """
    name_data = get_name_tf_matrices(dataset, song_name, feature, subdivision)
    return np.load(f"{paths.path_data_persisted}/{dataset}/barwise_tf/{name_data}.npy", allow_pickle = True)

def load_or_save_barwise_tf(dataset, song_path, bars, feature = "log_mel_grill", subdivision = 96, hop_length = 32):
    """
    Load the barwiseTF matrix (already pre-computed) for this song.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        return load_barwise_tf(dataset, song_name, feature, subdivision)
    
    except FileNotFoundError:
        spectrogram = load_or_save_spectrogram(dataset, song_path, feature, hop_length = hop_length)
        barwise_tf = bi.barwise_TF_matrix(spectrogram, bars, hop_length/44100, subdivision)
        name_data = get_name_tf_matrices(dataset, song_name, feature, subdivision)
        np.save(f"{paths.path_data_persisted}/{dataset}/barwise_tf/{name_data}", barwise_tf)
        return barwise_tf

def load_beatwise_tf(dataset, song_name, feature = "log_mel_grill", subdivision_beat = 24):
    """
    Load the beatwiseTF matrix (already pre-computed) for this song.
    """
    name_data = get_name_tf_matrices(dataset, song_name, feature, subdivision_beat)
    return np.load(f"{paths.path_data_persisted}/{dataset}/beatwise_tf/{name_data}.npy", allow_pickle = True)

def load_or_save_beatwise_tf(dataset, song_path, beats, feature = "log_mel_grill", subdivision_beat = 24, hop_length=32):
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        return load_beatwise_tf(dataset, song_name, feature, subdivision_beat)
    
    except FileNotFoundError:
        spectrogram = load_or_save_spectrogram(dataset, song_path, feature, hop_length = hop_length)
        beatwise_tf = bi.barwise_TF_matrix(spectrogram, beats, hop_length/44100, subdivision_beat)
        name_data = get_name_tf_matrices(dataset, song_name, feature, subdivision_beat)
        np.save(f"{paths.path_data_persisted}/{dataset}/beatwise_tf/{name_data}", beatwise_tf)
        return beatwise_tf
    
# %% Loading SSM - useful for TISMIR recomputation
def load_barwise_tf_ssm(dataset, song_name, feature = "log_mel_grill", subdivision = 96, similarity_type = "cosine", train = False):
    if dataset == "salami":
        if train:
            subset = "train"
        else:
            subset = "test"
        return np.load(f"{paths.path_data_persisted}/salami/self_similarity_matrices/{subset}/ssm_{dataset}_song{song_name}_barwiseTF_{similarity_type}_{feature}_subdiv{subdivision}.npy", allow_pickle = True)
    elif dataset == "rwcpop":
        return np.load(f"{paths.path_data_persisted}/rwcpop/self_similarity_matrices/ssm_{dataset}_song{song_name}_barwiseTF_{similarity_type}_{feature}_subdiv{subdivision}.npy", allow_pickle = True)
    else:
        raise err.InvalidArgumentValueException("Invalid dataset name, should be either 'salami' or 'rwcpop'.")

def load_beatwise_tf_ssm(dataset, song_name, feature = "log_mel_grill", subdivision = 96, similarity_type = "cosine", train = False):
    if dataset == "salami":
        if train:
            subset = "train"
        else:
            subset = "test"
        return np.load(f"{paths.path_data_persisted}/salami/self_similarity_matrices/{subset}/ssm_{dataset}_song{song_name}_beatwiseTF_{similarity_type}_{feature}_subdiv{subdivision}.npy", allow_pickle = True)
    elif dataset == "rwcpop":
        return np.load(f"{paths.path_data_persisted}/rwcpop/self_similarity_matrices/ssm_{dataset}_song{song_name}_beatwiseTF_{similarity_type}_{feature}_subdiv{subdivision}.npy", allow_pickle = True)
    else:
        raise err.InvalidArgumentValueException("Invalid dataset name, should be either 'salami' or 'rwcpop'.")


def load_beat_sync_ssm_foote(dataset, song_name):
    """
    Load the files for the beat-sync SSM of this song, computed with the Foote particular hyperparameters (as implemented in the MSAF toolbox).
    """
    ssm = np.load(f"{paths.path_data_persisted}/{dataset}/foote_experiments/self_similarity_matrices/ssm_{dataset}_song{song_name}_beatsync.npy", allow_pickle = True)
    beat_sync_times, duration = np.load(f"{paths.path_data_persisted}/{dataset}/foote_experiments/sync_times/synctimes_{dataset}_song{song_name}_beatsync.npy", allow_pickle = True)
    return ssm, beat_sync_times, duration

def load_bar_sync_ssm_foote(dataset, song_name):
    """
    Load the files for the bar-sync SSM of this song, computed with the Foote particular hyperparameters (as implemented in the MSAF toolbox).
    """
    ssm = np.load(f"{paths.path_data_persisted}/{dataset}/foote_experiments/self_similarity_matrices/ssm_{dataset}_song{song_name}_barsync.npy", allow_pickle = True)
    bar_sync_times, duration = np.load(f"{paths.path_data_persisted}/{dataset}/foote_experiments/sync_times/synctimes_{dataset}_song{song_name}_barsync.npy", allow_pickle = True)
    return ssm, bar_sync_times, duration

def load_barwise_tf_ssm_foote(dataset, song_name, subdivision = 96):
    """
    Load the barwiseTF SSM for this song, which was persisted after a first computation, or compute it if it wasn't found.
    """
    return np.load(f"{paths.path_data_persisted}/{dataset}/foote_experiments/self_similarity_matrices/ssm_{dataset}_song{song_name}_barwiseTF_subdiv{subdivision}.npy", allow_pickle = True)

# %% Annotation loaders
def load_mirex10_annot_song_RWC(song_number):
    """
    Loads the MIREX10 annotations for this song.
    """
    annotations_mirex = f"{paths.path_annotation_rwcpop}/MIREX10"
    annot_path_mirex = f"{annotations_mirex}/{dm.get_annotation_name_from_song(song_number, 'MIREX10')}"
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return references_segments

def load_RWC_dataset(music_folder_path, annotations_type = "MIREX10"):
    """
    Load the data on the RWC dataset, ie path of songs and annotations.
    The annotations can be either AIST or MIREX 10.

    Parameters
    ----------
    music_folder_path : String
        Path of the folder to parse.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        The default is "MIREX10"

    Raises
    ------
    NotImplementedError
        If the format is not taken in account.

    Returns
    -------
    numpy array
        list of list of paths, each sublist being of the form [song, annotations, downbeat(if specified)].
        
    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    # Load dataset paths at the format "song, annotations, downbeats"
    paths = []
    for file in os.listdir(music_folder_path):
        if file[-4:] == ".wav":
            file_number = "{:03d}".format(int(file[:-4]))
            ann = dm.get_annotation_name_from_song(file_number, annotations_type)
            paths.append([file, ann])
    return np.array(paths)





# %% tensor spectorgrams (not so much used, may be bugged)
def load_or_save_tensor_spectrogram(dataset, song_path, feature, hop_length, subdivision_bars, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Loads the BTF (bars - time - frequency) tensor for this song, which was persisted after a first computation, or compute it if it wasn't found.

    You should prefer load_or_save_spectrogram, as it allows more possibility about tensor folding, except if you're short in space on your disk.

    Parameters
    ----------
    dataset : string
        Name of the dataset. Only "rwcpop" and "salami" for now.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.
    n_fft and n_mfcc : integers, optional
        Both arguments are used respectively for the stft and for the mfcc computation, and are used to 

    Returns
    -------
    numpy array
        The tensor spectrogram of this song.

    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        tensor_barwise = np.load(f"{paths.path_data_persisted}/{dataset}/tensor_barwise_ae/{song_name}_{feature}_hop{hop_length}_subdiv{subdivision_bars}.npy", allow_pickle = True)
        return tensor_barwise
    except FileNotFoundError:
        the_signal, sampling_rate = librosa.load(song_path, sr=44100)
        raise NotImplementedError("Do not compute please")
        if "stft" in feature and "nfft" in feature:
            if "nfft" not in feature: 
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft)
            else:              
                n_fft_arg = int(feature.split("nfft")[1])
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft_arg)
        elif "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)
            else:
                n_mfcc_arg = int(feature.split("nmfcc")[1])
                spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
        else:
            spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
            
        hop_length_seconds = hop_length/44100
        bars = load_or_save_bars(persisted_path, song_path)
        tensor_spectrogram = nn_utils.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision_bars)
        np.save(f"{persisted_path}/tensor_barwise_ae/{song_name}_{feature}_hop{hop_length}_subdiv{subdivision_bars}", tensor_spectrogram)
        return tensor_spectrogram

def load_or_save_pcp_msaf(dataset, song_path, hop_length):
    """
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        pcp, frame_times, duration = np.load(f"{paths.path_data_persisted}/{dataset}/pcp_msaf/{song_name}_hop{hop_length}.npy", allow_pickle = True)
    except FileNotFoundError:
        signal, sr = librosa.load(song_path, sr=44100)
        duration = len(signal) / float(sr)
        pcp, frame_times = signal_to_spectrogram.get_pcp_as_msaf(signal, sr, hop_length)
        np.save(f"{paths.path_data_persisted}/{dataset}/pcp_msaf/{song_name}_hop{hop_length}", (pcp, frame_times, duration))

    return pcp, frame_times, duration
