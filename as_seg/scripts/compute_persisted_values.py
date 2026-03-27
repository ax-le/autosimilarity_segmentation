# -*- coding: utf-8 -*-
import as_seg.scripts.default_path as paths
import as_seg.model.errors as err
import as_seg.scripts.overall_scripts as scr
import as_seg.data_manipulation as dm
import as_seg.CBM_algorithm as CBM
import as_seg.autosimilarity_computation as as_comp
import as_seg.barwise_input as bi
import as_seg.model.signal_to_spectrogram as signal_to_spectrogram
from as_seg.model.commmon_plot import *

import math
import numpy as np
import pandas as pd
import mirdata
import os
import tensorly as tl
import soundfile as sf
import librosa

bands_numbers = [None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def get_scores_different_tolerance(tol, segments, bars, ref_tab):
    res = math.inf * np.ones((2, 3))
    
    # Tolerance in absolute time
    if tol == "s":
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
        
        prec05, rap05, f_mes05 = dm.compute_score_of_segmentation(ref_tab[0], segments_in_time, window_length = 0.5)
        prec3, rap3, f_mes3 = dm.compute_score_of_segmentation(ref_tab[0], segments_in_time, window_length = 3)
        res = [[round(prec05,4),round(rap05,4),round(f_mes05,4)], [round(prec3,4),round(rap3,4),round(f_mes3,4)]]
        
        if len(ref_tab) > 1:
            nd_prec05, nd_rap05, nd_f_mes05 = dm.compute_score_of_segmentation(ref_tab[1], segments_in_time, window_length = 0.5)
            nd_prec3, nd_rap3, nd_f_mes3 = dm.compute_score_of_segmentation(ref_tab[1], segments_in_time, window_length = 3)
            if nd_f_mes05 + nd_f_mes3 > f_mes05 + f_mes3:
                res = [[round(nd_prec05,4),round(nd_rap05,4),round(nd_f_mes05,4)], [round(nd_prec3,4),round(nd_rap3,4),round(nd_f_mes3,4)]]
    
    # Tolerance barwise aligned
    elif tol == "b":
        raise NotImplementedError("Beatwise Only here")
        ref0_in_bars = np.array(dm.segments_from_time_to_bar(ref_tab[0], bars))
        
        prec0bar, rap0bar, f_mes0bar = dm.compute_score_of_segmentation(ref0_in_bars, segments, window_length = 0)
        prec1bar, rap1bar, f_mes1bar = dm.compute_score_of_segmentation(ref0_in_bars, segments, window_length = 1)
        res = [[round(prec0bar,4),round(rap0bar,4),round(f_mes0bar,4)], [round(prec1bar,4),round(rap1bar,4),round(f_mes1bar,4)]]

        if len(ref_tab) > 1:
            ref1_in_bars = np.array(dm.segments_from_time_to_bar(ref_tab[1], bars))

            nd_prec0bar, nd_rap0bar, nd_f_mes0bar = dm.compute_score_of_segmentation(ref1_in_bars, segments, window_length = 0)
            nd_prec1bar, nd_rap1bar, nd_f_mes1bar = dm.compute_score_of_segmentation(ref1_in_bars, segments, window_length = 1)
            if nd_f_mes0bar + nd_f_mes1bar > f_mes0bar + f_mes1bar:
                res = [[round(nd_prec0bar,4),round(nd_rap0bar,4),round(nd_f_mes0bar,4)], [round(nd_prec1bar,4),round(nd_rap1bar,4),round(nd_f_mes1bar,4)]]
        
    else:
        raise NotImplementedError("Tol not understood")
    
    return res

def load_or_save_beatwise_tf(song_path, persisted_path, beats, subdivision_beat):
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        beatwise_tf = np.load(f"{persisted_path}/beatwise_tf/{song_name}_log_mel_grill_hop32_subdiv{subdivision_beat}.npy", allow_pickle = True)
        return beatwise_tf
    
    except FileNotFoundError:
        the_signal, sampling_rate = librosa.load(song_path, sr=44100)
        #the_signal, sampling_rate = sf.read(song_path)
        spectrogram = signal_to_spectrogram.get_spectrogram(the_signal, sampling_rate, "log_mel_grill", hop_length = 32)
        beatwise_tf = bi.barwise_TF_matrix(spectrogram, beats, 32/44100, subdivision_beat)
        np.save(f"{persisted_path}/beatwise_tf/{song_name}_log_mel_grill_hop32_subdiv{subdivision_beat}", beatwise_tf)
        return beatwise_tf

def compute_beat_and_beatwisetf(feature):
    salami = mirdata.initialize('salami', data_home = paths.path_entire_salami)
    len_salami = len(salami.track_ids)
    subdivision_beat = 24
    hop_length = 32
    hop_length_seconds = hop_length/44100

    all_tracks = salami.load_tracks()
    
    song_idx = 0
           
    for key, track in all_tracks.items():
        try:
            print(key)
            beats = scr.load_or_save_beats(paths.path_data_persisted_salami, track.audio_path)
            beatwise_tf = load_or_save_beatwise_tf(track.audio_path, paths.path_data_persisted_salami, beats, subdivision_beat)

        except FileNotFoundError:
            print(f"{key} not found, normal ?")            
            
        except MemoryError:
            print(f"{key} too large")
        
        except err.ToDebugException:
            print(f"{key}: duplicate samples when computing the beatwise TF matrix")

def compute_pcp_msaf_salami():
    salami = mirdata.initialize('salami', data_home = paths.path_entire_salami)
    hop_length_barwise = 64
    hop_length_feature_scale = 1024

    all_tracks = salami.load_tracks()
    
    for key, track in all_tracks.items():
        try:
            print(key)
            #pcp, frame_times, duration = scr.load_or_save_pcp_msaf(paths.path_data_persisted_salami, track.audio_path, hop_length_feature_scale)
            scr.load_or_save_pcp_msaf(paths.path_data_persisted_salami, track.audio_path, hop_length_barwise)
            #oversampled_pcp, _, _ = scr.load_or_save_pcp_msaf(paths.path_data_persisted_salami, track.audio_path, hop_length_barwise)

        except FileNotFoundError:
            print(f"{key} not found, normal ?")            
            
        except MemoryError:
            print(f"{key} too large")

def compute_pcp_msaf_rwc():
    hop_length_barwise = 64
    hop_length_feature_scale = 1024

    for song_name in range(1,101):
        print(song_name)
        song_path = f"{paths.path_entire_rwc}/{song_name}.wav"
        scr.load_or_save_pcp_msaf(paths.path_data_persisted_rwc, song_path, hop_length_feature_scale)
        scr.load_or_save_pcp_msaf(paths.path_data_persisted_rwc, song_path, hop_length_barwise)
        #oversampled_pcp, _, _ = scr.load_or_save_pcp_msaf(paths.path_data_persisted_salami, track.audio_path, hop_length_barwise)

 
#if __name__ == "__main__":
compute_pcp_msaf_rwc()
print("Done.")