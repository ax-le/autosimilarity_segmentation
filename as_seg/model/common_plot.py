# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:29:17 2019

@author: amarmore

Defining common plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from base_audio.common_plot import *

# %% Plotting utils
def plot_lenghts_hist(lengths):
    """
    Plots the lengths of segments in an histogram
    i.e. the distribution of the size of segments in the annotation/estimation 
    (allegedly already computed into a list lengths).
 
    Parameters
    ----------
    lengths : list of integers
        List of all segments' sizes in the annotation/estimation.
    """
    plt.rcParams.update({'font.size': 18})

    fig, axs = plt.subplots(1, 1, figsize=(6, 3.75))
    axs.hist(lengths, bins = range(1,34), density = True, cumulative = False, align = "left")
    plt.xticks(np.concatenate([[1],range(4, 34, 4)]))
    plt.ylim(0,1)

    axs.set_xlabel("Size of the segment,\nin number of bars")
    axs.set_ylabel("Proportion among\nall segments")

    plt.show()

def plot_measure_with_annotations(measure, annotations, color = "red"):
    """
    Plots the measure (typically novelty) with the segmentation annotation.
    """
    plt.plot(np.arange(len(measure)),measure, color = "black")
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_measure_with_annotations_and_prediction(measure, annotations, frontiers_predictions, title = "Title"):
    """
    Plots the measure (typically novelty) with the segmentation annotation and the estimated segmentation.
    """
    plt.title(title)
    plt.plot(np.arange(len(measure)),measure, color = "black")
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "red")
    for x in frontiers_predictions:
        if x in annotations:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_segments_with_annotations(seg, annot):
    """
    Plots the estimated labelling of segments next to with the frontiers in the annotation.
    """
    for x in seg:
        plt.plot([x[0], x[1]], [x[2]/10,x[2]/10], '-', linewidth=1, color = "black")
    for x in annot:
        plt.plot([x[1], x[1]], [0,np.amax(np.array(seg)[:,2])/10], '-', linewidth=1, color = "red")
    plt.show()
    