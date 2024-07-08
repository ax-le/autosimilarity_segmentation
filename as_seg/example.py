# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2023

@author: amarmore

A test sample for the CBM algorithm.
"""

# External imports
# Module for manipulating arrays
import numpy as np

# Module for reading signals
import librosa

# Module to handle annotations for MIR files
import mirdata

# Module encapsulating the computation of features from the signal
import as_seg.model.signal_to_spectrogram as signal_to_spectrogram

# General module for manipulating data: conversion between time, bars, frame indexes, loading of data, ...
import as_seg.data_manipulation as dm

# Module to process the input in barwise features
import as_seg.barwise_input as bi

# Module to process the compute the autosimilarity
import as_seg.autosimilarity_computation as as_computation

# Module containing the CBM algorithm
import as_seg.CBM_algorithm as cbm

# Plotting module
from as_seg.model.common_plot import *

# %% Loading annotations and defining the audio path
path_to_beatles_dataset = '/home/a23marmo/datasets/beatles' # To change
beatles = mirdata.initialize('beatles', path_to_beatles_dataset)
beatles.download()

# NB: you have to place the file "01_-_Come_Together.wav" manually in the folder "audio/11_-_Abbey_Road/"
come_together = beatles.track('1101')

references_segments = beatles.load_sections(come_together.sections_path).intervals

song_path = come_together.audio_path

# %% Loading the song, as signal, and estimate downbeats
the_signal, sampling_rate = librosa.load(song_path, sr = None)
bars = dm.get_bars_from_audio(song_path)

# %% Computing the features
hop_length = 32 # Oversampling the spectrogram, to select frames which will be equally-spaced barwise.
hop_length_seconds = hop_length/sampling_rate # As bars are in seconds, we convert this hop length in seconds.
subdivision_bars = 96 # The number of time samples to consider in each bar.

feature_object = signal_to_spectrogram.FeatureObject(sr=sampling_rate, feature="log_mel", hop_length=hop_length, mel_grill=True)
log_mel = feature_object.get_spectrogram(the_signal) # Log_mel spectrogram

barwise_TF = bi.barwise_TF_matrix(log_mel, bars, hop_length_seconds, subdivision_bars)

# %% Cosine autosimilarity
barwise_TF_cosine_autosimilarity = as_computation.switch_autosimilarity(barwise_TF, "cosine")
#Alternatively, one could use: as_computation.get_cosine_autosimilarity(barwise_TF_cosine)
plot_me_this_spectrogram(barwise_TF_cosine_autosimilarity, title = "Cosine autosimilarity of the Barwise TF matrix")

#%% Running the CBM on the autosimilarity matrix
segments_cbm_cosine = cbm.compute_cbm(barwise_TF_cosine_autosimilarity, penalty_weight = 1, penalty_func = "modulo8", bands_number = 7)[0]
segments_cbm_cosine_in_time = dm.segments_from_bar_to_time(segments_cbm_cosine, bars)

score_cbm_cosine_zero_five = dm.compute_score_of_segmentation(references_segments, segments_cbm_cosine_in_time, window_length = 0.5)
print(f"Score with 0.5 second tolerance: Precision {score_cbm_cosine_zero_five[0]}, Recall {score_cbm_cosine_zero_five[1]}, F measure {score_cbm_cosine_zero_five[2]}")
score_cbm_cosine_three = dm.compute_score_of_segmentation(references_segments, segments_cbm_cosine_in_time, window_length = 3)
print(f"Score with 3 seconds tolerance: Precision {score_cbm_cosine_three[0]}, Recall {score_cbm_cosine_three[1]}, F measure {score_cbm_cosine_three[2]}")

# %% Autocorrelation/Covariance autosimilarity
barwise_TF_covariance_autosimilarity = as_computation.switch_autosimilarity(barwise_TF, "covariance")
plot_me_this_spectrogram(barwise_TF_covariance_autosimilarity, title = "Covariance autosimilarity of the Barwise TF matrix")

# %% Running the CBM on the autosimilarity matrix
segments_cbm_covariance = cbm.compute_cbm(barwise_TF_covariance_autosimilarity, penalty_weight = 1, penalty_func = "modulo8", bands_number = 7)[0]
segments_cbm_covariance_in_time = dm.segments_from_bar_to_time(segments_cbm_covariance, bars)

score_cbm_covariance_zero_five = dm.compute_score_of_segmentation(references_segments, segments_cbm_covariance_in_time, window_length = 0.5)
print(f"Score with 0.5 second tolerance: Precision {score_cbm_covariance_zero_five[0]}, Recall {score_cbm_covariance_zero_five[1]}, F measure {score_cbm_covariance_zero_five[2]}")
score_cbm_covariance_three = dm.compute_score_of_segmentation(references_segments, segments_cbm_covariance_in_time, window_length = 3)
print(f"Score with 3 seconds tolerance: Precision {score_cbm_covariance_three[0]}, Recall {score_cbm_covariance_three[1]}, F measure {score_cbm_covariance_three[2]}")

# %% RBF autosimilarity
barwise_TF_rbf_autosimilarity = as_computation.switch_autosimilarity(barwise_TF, "RBF")
plot_me_this_spectrogram(barwise_TF_rbf_autosimilarity, title = "RBF autosimilarity of the Barwise TF matrix")

# %% Running the CBM on the autosimilarity matrix
segments_cbm_rbf = cbm.compute_cbm(barwise_TF_rbf_autosimilarity, penalty_weight = 1, penalty_func = "modulo8", bands_number = 7)[0]
segments_cbm_rbf_in_time = dm.segments_from_bar_to_time(segments_cbm_rbf, bars)

score_cbm_rbf_zero_five = dm.compute_score_of_segmentation(references_segments, segments_cbm_rbf_in_time, window_length = 0.5)
print(f"Score with 0.5 second tolerance: Precision {score_cbm_rbf_zero_five[0]}, Recall {score_cbm_rbf_zero_five[1]}, F measure {score_cbm_rbf_zero_five[2]}")
score_cbm_rbf_three = dm.compute_score_of_segmentation(references_segments, segments_cbm_rbf_in_time, window_length = 3)
print(f"Score with 3 seconds tolerance: Precision {score_cbm_rbf_three[0]}, Recall {score_cbm_rbf_three[1]}, F measure {score_cbm_rbf_three[2]}")

# %% Conclusion
import pandas as pd
columns = np.array(["Precision 0.5", "Recall 0.5", "F measure 0.5","Precision 3", "Recall 3", "F measure 3"])  
tab = []
tab.append([round(score_cbm_cosine_zero_five[0],5), round(score_cbm_cosine_zero_five[1],5),round(score_cbm_cosine_zero_five[2],5),round(score_cbm_cosine_three[0],5),round(score_cbm_cosine_three[1],5),round(score_cbm_cosine_three[2],5)])
tab.append([round(score_cbm_covariance_zero_five[0],5), round(score_cbm_covariance_zero_five[1],5),round(score_cbm_covariance_zero_five[2],5),round(score_cbm_covariance_three[0],5),round(score_cbm_covariance_three[1],5),round(score_cbm_covariance_three[2],5)])
tab.append([round(score_cbm_rbf_zero_five[0],5), round(score_cbm_rbf_zero_five[1],5),round(score_cbm_rbf_zero_five[2],5),round(score_cbm_rbf_three[0],5),round(score_cbm_rbf_three[1],5),round(score_cbm_rbf_three[2],5)])

pd.DataFrame(tab, index=["Cosine", "Covariance", "RBF"], columns=columns)