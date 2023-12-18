# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:34:39 2020

@author: amarmore

A module containing some high-level scripts for decomposition and/or segmentation.
Not meant to be shared but rather to work.
"""

import soundfile as sf
import librosa.core
import librosa.feature
import librosa
import os
import numpy as np

import as_seg.data_manipulation as dm
import as_seg.barwise_input as bi
import as_seg.model.features as features
import as_seg.model.errors as err
import as_seg.scripts.default_path as paths

def load_spec_annot_song_RWC(song_number, feature, hop_length = 32):
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
    spectrogram = load_or_save_spectrogram(paths.path_data_persisted_rwc, f"{paths.path_entire_rwc}/{song_number}.wav", feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_or_save_bars(paths.path_data_persisted_rwc, f"{paths.path_entire_rwc}/{song_number}.wav")
    annotations_mirex = f"{paths.path_annotation_rwc}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

def load_bar_annot_song_RWC(song_number, hop_length = 32):
    """
    Similar to load_spec_annot_song_RWC(), but without loading the soectrogram 
    (only bars and annotations).
    
    Generally used when spectrogram is not desired but has a huge computation or memory cost.
    """
    bars = load_or_save_bars(paths.path_data_persisted_rwc, f"{paths.path_entire_rwc}/{song_number}.wav")
    annotations_mirex = f"{paths.path_annotation_rwc}/MIREX10"
    annot_path_mirex = "{}/{}".format(annotations_mirex, dm.get_annotation_name_from_song(song_number, "MIREX10"))
    annotations = dm.get_segmentation_from_txt(annot_path_mirex, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return bars, references_segments
    
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
    spectrogram = load_or_save_spectrogram(paths.path_data_persisted_come_together, f"{paths.come_together}.wav", feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20)
    bars = load_or_save_bars(paths.path_data_persisted_come_together, f"{paths.come_together}.wav")
    annotation_path = f"{paths.come_together}.lab"
    annotations = dm.get_segmentation_from_txt(annotation_path, "MIREX10")
    references_segments = np.array(annotations)[:,0:2]
    return spectrogram, bars, references_segments

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

# %% Loading or persisting bars, spectrograms and already computed neural networks latent projections
def load_or_save_bars(persisted_path, song_path):
    """
    Computes the bars for this song, or load them if they were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_path : string
        The path of the signal of the song.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    except:
        bars = dm.get_bars_from_audio(song_path)
        np.save("{}/bars/{}".format(persisted_path, song_name), bars)
    return bars

def load_bars(persisted_path, song_name):
    """
    Loads the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_name : string
        Name of the song (identifier of the bars to load).

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    raise err.OutdatedBehaviorException("You should use load_or_save_bars(persisted_path, song_path) instead, as it handle the fact that bars weren't computed yet.")
    bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    return bars
    
def load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
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
        if "stft" in feature:   
            if "nfft" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nfft{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_fft, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "mel" or feature == "log_mel":
            raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
        elif "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_mfcc, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "pcp":
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}_{}.npy".format(persisted_path, song_name, feature, hop_length, fmin))
        else:
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))

    except FileNotFoundError:
        if "log_mel_grill" in feature:
            mel = load_or_save_spectrogram(persisted_path, song_path, "mel_grill", hop_length, fmin = fmin, n_fft = n_fft, n_mfcc = n_mfcc)
            return features.get_log_mel_from_mel(mel, feature)
        else:
            the_signal, original_sampling_rate = sf.read(song_path)
            if original_sampling_rate != 44100:
                the_signal = librosa.core.resample(np.asfortranarray(the_signal), original_sampling_rate, 44100)
            if "stft" in feature:
                if "nfft" not in feature: 
                    spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft)
                    np.save("{}/spectrograms/{}_{}-nfft{}_stereo_{}".format(persisted_path, song_name, feature, n_fft, hop_length), spectrogram)
                    return spectrogram
                else:              
                    n_fft_arg = int(feature.split("nfft")[1])
                    spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft_arg)
                    np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                    return spectrogram
            if feature == "mel" or feature == "log_mel":
                raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
            if "mfcc" in feature:
                if "nmfcc" not in feature:
                    spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)
                    np.save("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}".format(persisted_path, song_name, feature, n_mfcc, hop_length), spectrogram)
                    return spectrogram        
                else:
                    n_mfcc_arg = int(feature.split("nmfcc")[1])
                    spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
                    np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                    return spectrogram
            if feature == "pcp_tonnetz":
                # If chromas are already computed, try to load them instead of recomputing them.
                chromas = load_or_save_spectrogram(persisted_path, song_path, "pcp", hop_length, fmin = fmin)
                spectrogram = librosa.feature.tonnetz(y=None, sr = None, chroma = chromas)
                np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
                return spectrogram
            # if feature == "tonnetz":
            #     hop_length = "fixed"
            #     fmin = "fixed"
            if feature == "pcp":
                # If it wasn't pcp_tonnetz, compute the spectrogram, and then save it.
                spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
                np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
                return spectrogram
        
            spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length)
            np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
            return spectrogram

    return spectrogram

def load_or_save_spectrogram_and_bars(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Loads the spectrogram and the bars for this song, which were persisted after a first computation, or compute them if they weren't found.

    Parameters
    ----------
    persisted_path : string
        Path where the bars and the spectrogram should be found.
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
    bars = load_or_save_bars(persisted_path, song_path)
    spectrogram = load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = fmin, n_fft = n_fft, n_mfcc = n_mfcc)
    return bars, spectrogram
