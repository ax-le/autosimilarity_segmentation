# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:09:45 2021

@author: amarmore
"""

from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import pandas as pd
pd.set_option('precision', 4)
import matplotlib.colors as colors
import matplotlib.cm as cm

import musicntd.autosimilarity_segmentation as as_seg
import musicntd.data_manipulation as dm
import musicntd.tensor_factory as tf
import musicntd.scripts.overall_scripts as scr
from musicntd.model.current_plot import *
import temporalclustering.tv_kmeans as tvk

annotations_folder_path = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\annotations"
persisted_path = "C:\\Users\\amarmore\\Desktop\\data_persisted"
entire_rwc = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\Entire RWC"

# %% Dymanic programming with clustering cost
def dynamic_clustering_cost_computation(autosimilarity, penalty_weight, penalty_func):
    """
    TODO
    """
    costs = [-inf for i in range(len(autosimilarity))]
    segments_best_ends = [None for i in range(len(autosimilarity))]
    segments_best_ends[0] = 0
    costs[0] = 0
    kmeans = KMeans(n_clusters=10, random_state=0).fit(autosimilarity)
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in as_seg.possible_segment_start(current_idx, min_size = 2, max_size = 36):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")

            clus_cost = 1/len(np.unique(kmeans.labels_[possible_start_idx:current_idx]))**2
            
            segment_length = current_idx - possible_start_idx
            penalty_cost = as_seg.penalty_cost_from_arg(penalty_func, segment_length)            
            
            this_segment_cost = clus_cost * segment_length - penalty_weight * penalty_cost

            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost > costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_ends[current_idx] = 0
            else:
                if costs[possible_start_idx] + this_segment_cost > costs[current_idx]:
                    costs[current_idx] = costs[possible_start_idx] + this_segment_cost
                    segments_best_ends[current_idx] = possible_start_idx

    segments = [(segments_best_ends[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_end = segments_best_ends[len(autosimilarity) - 1]
    while precedent_end > 0:
        segments.append((segments_best_ends[precedent_end], precedent_end))
        precedent_end = segments_best_ends[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], costs[-1]

# %% 
def Q_with_noise(q_factor, std):
    m,n = q_factor.shape
    noise = np.abs(np.random.normal(0, std, m*n)).reshape(m,n)
    return q_factor + noise

def this_song_Q_with_noise(song_number, ranks, std_vals, draws_nb, convolution_type = "eight_bands", penalty_func = "modulo8"):
    """
    Segmentation results when ranks and penalty parameter are fitted by cross validation.
    Results are shown for the test dataset.
    """
    hop_length = 32
    hop_length_seconds = 32/44100
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    subdivision = 96
    penalty_weight = 0
    dataset_paths = scr.load_RWC_dataset(entire_rwc, annotations_type)
    
    all_res = -1 * np.ones((len(std_vals), draws_nb, 2, 1))
    
    annot_path = "{}\\{}".format(annotations_folder, dm.get_annotation_name_from_song(song_number, annotations_type))
    annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
    references_segments = np.array(annotations)[:, 0:2]
    
    bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(entire_rwc, song_number), "pcp", hop_length)
            
    tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
    
    persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
    q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
    autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
    baseline_segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
    nb_unique_segs = np.ones(len(std_vals))
    for idx_std, std in enumerate(std_vals):
        all_segs = []
        for i in range(draws_nb):

            q_fac_noised = Q_with_noise(q_factor, std)
            autosimilarity = as_seg.get_autosimilarity(q_fac_noised, transpose = True, normalize = True)
                
            segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
            all_segs.append(segments)
            prec, rap, f_mes = dm.compute_score_of_segmentation(np.array(baseline_segments), np.array(segments), window_length = 0.5)
            all_res[idx_std, i, 0] = round(f_mes,4)
            count = 0
            for seg in segments:
                if seg not in baseline_segments:
                    count += 1
            all_res[idx_std, i, 1] = count
        found = []
        for a_seg in all_segs:
            if a_seg not in found:
                found.append(a_seg)
        nb_unique_segs[idx_std] = len(found)
            
    avg_fone_for_stds = -1 * np.ones((len(std_vals), 2))
    for std_idx in range(len(std_vals)):
        avg_fone_for_stds[std_idx, 0] = np.mean(all_res[std_idx, :, 0, 0])
        avg_fone_for_stds[std_idx, 1] = np.std(all_res[std_idx, :, 0, 0])
        
    plt.errorbar(std_vals, avg_fone_for_stds[:,0], avg_fone_for_stds[:,1], linestyle='None', marker='^')
    #plt.errorbar(rank_rhythm, avg_stds_for_h[:,1,0], avg_stds_for_h[:,1,1], linestyle='None', marker='^')
    plt.xlabel("Std of noise")
    plt.ylabel("F measure (compared to seg without noise)")
    plt.title("Variation of the F measure\nof segmentation with nd without noise\n when varying the value of the std")
    plt.show()
    
    plt.plot(std_vals, nb_unique_segs, linestyle='None', marker='^')
    plt.xlabel("Std of noise")
    plt.ylabel("Nombre de segmentations uniques")
    plt.title("Count of unique segmentation in function of the noise")
    plt.show()
        
    
    avg_val_dif_for_stds = -1 * np.ones((len(std_vals), 2))
    for std_idx in range(len(std_vals)):
        avg_val_dif_for_stds[std_idx, 0] = np.mean(all_res[std_idx, :, 1, 0])
        avg_val_dif_for_stds[std_idx, 1] = np.std(all_res[std_idx, :, 1, 0])
        

    plt.errorbar(std_vals, avg_val_dif_for_stds[:,0], avg_val_dif_for_stds[:,1], linestyle='None', marker='^')
    #plt.errorbar(rank_rhythm, avg_stds_for_h[:,1,0], avg_stds_for_h[:,1,1], linestyle='None', marker='^')
    plt.xlabel("Std of noise")
    plt.ylabel("F measure (compared to seg without noise)")
    plt.title("Count of new frontiers in noisy one")
    plt.show()
    
        


    

# %% Clustering
def script_cluster_this_song(song_number, ranks, nb_clusters = 4, lamb = 20, mu = 20, iter_clus = 10):
    hop_length = 32
    hop_length_seconds = 32/44100
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    subdivision = 96
    
    annot_path = "{}\\{}".format(annotations_folder, dm.get_annotation_name_from_song(song_number, annotations_type))
    annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
    references_segments = np.array(annotations)[:, 0:2]
    
    bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(entire_rwc, song_number), "pcp", hop_length)
            
    tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
    
    persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)

    q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
    
    autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
    segments_dyn_prog = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = 1, penalty_func = "modulo8", convolution_type = "mixed")[0]
    segments_in_time_dyn_prog = dm.segments_from_bar_to_time(segments_dyn_prog, bars)
    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time_dyn_prog, window_length = 0.5)
    print(prec, rap, f_mes)
    
    custom_sigma = np.zeros(len(autosimilarity))
    for idx, couple in enumerate(segments_dyn_prog):
        custom_sigma[couple[0]:couple[1]] = int(idx)
    if segments_dyn_prog[-1][0] != segments_dyn_prog[-1][1]:
        custom_sigma[-1] = custom_sigma[-2]
    clusters = tvk.tv_kmeans(q_factor.T, int(custom_sigma[-1] + 1), lamb=lamb, mu=mu, itermax=100, init="custom", custom_sigma = custom_sigma)[1]  # TV regularized
    frontiers = []
    for i in range(len(clusters) - 1):
        if clusters[i] != clusters[i+1]:
            frontiers.append(i+1) # i + 1 car segment ne comprend pas dernière mesure (car la mesure d'après commence à la frontière)
    frontiers.append(len(clusters) - 1)
    
    #frontiers = frontiers_several_runs_clustering(q_factor, nb_clusters = nb_clusters, lamb = lamb, mu = mu, iter_clus = iter_clus)
    segments = dm.frontiers_to_segments(frontiers)
    segments_in_time = dm.segments_from_bar_to_time(segments, bars)
    
    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
    print(prec, rap, f_mes)

    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
    
    

def script_cluster_this_song_dyn_init_range_mu_lambda(song_number, ranks, lambda_range = range(1,20), mu_range = range(1,20)):
    hop_length = 32
    hop_length_seconds = 32/44100
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    subdivision = 96
    
    annot_path = "{}\\{}".format(annotations_folder, dm.get_annotation_name_from_song(song_number, annotations_type))
    annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
    references_segments = np.array(annotations)[:, 0:2]
    
    bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(entire_rwc, song_number), "pcp", hop_length)
            
    tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
    
    persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)

    q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
    
    autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
    segments_dyn_prog = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = 1, penalty_func = "modulo8", convolution_type = "mixed")[0]
    segments_in_time_dyn_prog = dm.segments_from_bar_to_time(segments_dyn_prog, bars)
    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time_dyn_prog, window_length = 0.5)
    f_mes_conv = f_mes
    
    custom_sigma = np.zeros(len(autosimilarity))
    for idx, couple in enumerate(segments_dyn_prog):
        custom_sigma[couple[0]:couple[1]] = int(idx)
    if segments_dyn_prog[-1][0] != segments_dyn_prog[-1][1]:
        custom_sigma[-1] = custom_sigma[-2]

    res = -1 * np.ones((len(lambda_range), len(mu_range)))
    for idx_lamb, lamb in enumerate(lambda_range):
        for idx_mu, mu in enumerate(mu_range):
            n_clus = int(custom_sigma[-1] + 1)
            clusters = tvk.tv_kmeans(q_factor.T, n_clus, lamb=lamb, mu=mu, itermax=100, init="custom", custom_sigma = custom_sigma)[1]  # TV regularized
            frontiers = []
            for i in range(len(clusters) - 1):
                if clusters[i] != clusters[i+1]:
                    frontiers.append(i+1) # i + 1 car segment ne comprend pas dernière mesure (car la mesure d'après commence à la frontière)
            frontiers.append(len(clusters) - 1)
            segments = dm.frontiers_to_segments(frontiers)
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
            
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            res[idx_lamb, idx_mu] = f_mes
        
    min_gap_pattern = min(min([lambda_range[i] - lambda_range[i-1] for i in range(1, len(lambda_range))]), min([mu_range[i] - mu_range[i-1] for i in range(1, len(mu_range))]))
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1,1, projection='3d')
    
    dz = res.flatten(order='F')
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float)/offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.jet(norm(fracs.tolist()))

    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.
    x_data, y_data = np.meshgrid(np.array(lambda_range),
                                 np.array(mu_range))
    
    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = res.T.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              min_gap_pattern, min_gap_pattern, z_data, color=color_values)
    ax.set_xlabel('Paramètre lambda')
    ax.set_ylabel('Paramètremu')
    ax.set_zlabel('F measure')
    ax.set_title('F measure for this song,\nfunction of mu and lambda parameters')
    # Finally, display the plot.
    plt.show()


def script_cluster_all_songs_dyn_init_range_mu_lambda(ranks, lambda_range = range(1,20), mu_range = range(1,20)):
    hop_length = 32
    hop_length_seconds = 32/44100
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    subdivision = 96
    dataset = entire_rwc
    dataset_paths = scr.load_RWC_dataset(dataset, annotations_type)
    f_mes_dyn_prog = 0
    all_res_songs = -1 * np.ones((100, len(lambda_range), len(mu_range), 1))
    
    for song_idx, song_and_annotations in enumerate(dataset_paths):
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(entire_rwc, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
    
        q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
        
        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
        segments_dyn_prog = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = 1, penalty_func = "modulo8", convolution_type = "mixed")[0]
        segments_in_time_dyn_prog = dm.segments_from_bar_to_time(segments_dyn_prog, bars)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time_dyn_prog, window_length = 0.5)
        f_mes_dyn_prog += f_mes
        
        custom_sigma = np.zeros(len(autosimilarity))
        for idx, couple in enumerate(segments_dyn_prog):
            custom_sigma[couple[0]:couple[1]] = int(idx)
        if segments_dyn_prog[-1][0] != segments_dyn_prog[-1][1]:
            custom_sigma[-1] = custom_sigma[-2]
    
        for idx_lamb, lamb in enumerate(lambda_range):
            for idx_mu, mu in enumerate(mu_range):
                n_clus = int(custom_sigma[-1] + 1)
                clusters = tvk.tv_kmeans(q_factor.T, n_clus, lamb=lamb, mu=mu, itermax=100, init="custom", custom_sigma = custom_sigma)[1]  # TV regularized
                frontiers = []
                for i in range(len(clusters) - 1):
                    if clusters[i] != clusters[i+1]:
                        frontiers.append(i+1) # i + 1 car segment ne comprend pas dernière mesure (car la mesure d'après commence à la frontière)
                frontiers.append(len(clusters) - 1)
                segments = dm.frontiers_to_segments(frontiers)
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                all_res_songs[song_idx, idx_lamb, idx_mu] = f_mes
    res = -1 * np.ones((len(lambda_range), len(mu_range), 1))
    f_mes_dyn_prog /= 100
    for idx_lamb, lamb in enumerate(lambda_range):
        for idx_mu, mu in enumerate(mu_range):
            res[idx_lamb, idx_mu] = np.mean(all_res_songs[:,idx_lamb, idx_mu]) - f_mes_dyn_prog
    min_gap_pattern = min(min([lambda_range[i] - lambda_range[i-1] for i in range(1, len(lambda_range))]), min([mu_range[i] - mu_range[i-1] for i in range(1, len(mu_range))]))
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1,1, projection='3d')
    
    dz = res.flatten(order='F')
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float)/offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.jet(norm(fracs.tolist()))

    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.
    x_data, y_data = np.meshgrid(np.array(lambda_range),
                                 np.array(mu_range))
    
    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = res.T.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              min_gap_pattern - min_gap_pattern/5, min_gap_pattern - min_gap_pattern/5, z_data, color=color_values)
    ax.set_xlabel('Paramètre lambda')
    ax.set_ylabel('Paramètremu')
    ax.set_zlabel('F measure')
    ax.set_title('F measure for RWC Pop, difference from dynamic prog one\nfunction of mu and lambda parameters')
    # Finally, display the plot.
    plt.show()
    
    return res


def this_song_res_clus_several_ranks(song_number, ranks_rhythm, ranks_pattern, nb_clusters = 4, lamb = 20, mu = 20, iter_clus = 10):
    """
    Segmentation results when ranks and penalty parameter are fitted by cross validation.
    Results are shown for the test dataset.
    """
    hop_length = 32
    hop_length_seconds = 32/44100
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    subdivision = 96

    all_res = -1 * np.ones((len(ranks_rhythm), len(ranks_pattern), 2, 6))
    
    annot_path = "{}\\{}".format(annotations_folder, dm.get_annotation_name_from_song(song_number, annotations_type))
    annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
    references_segments = np.array(annotations)[:, 0:2]
    
    bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(entire_rwc, song_number), "pcp", hop_length)
            
    tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
    
    for q_idx, rank_pattern in enumerate(ranks_pattern):
        for h_idx, rank_rhythm in enumerate(ranks_rhythm):
            ranks = [12, rank_rhythm, rank_pattern]
            persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
            q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
            frontiers = frontiers_several_runs_clustering(q_factor, nb_clusters = nb_clusters, lamb = lamb, mu = mu, iter_clus = iter_clus)
            segments = dm.frontiers_to_segments(frontiers)
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
            
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            all_res[h_idx, q_idx, 0] = [tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)]

            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
            all_res[h_idx, q_idx, 1] = [tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)]
    
    plot_a_song(all_res, ranks_rhythm, ranks_pattern)
    return all_res
