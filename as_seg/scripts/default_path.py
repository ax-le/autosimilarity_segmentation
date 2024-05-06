# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:22:59 2021

Default paths for the different folders.
Should be changed prior to any computation.

@author: amarmore
"""
import os

# We suppose that we are in the Notebooks folder,
# hence data is in the parent folder.
path_parent_of_data = os.path.dirname(os.getcwd())

# Global
path_data_persisted = f"{path_parent_of_data}/data/data_persisted"

# RWC
#path_data_persisted_rwcpop = "C:/Users/amarmore/Desktop/data_persisted/RWC Pop" ## Path where pre-computed data on RWC Pop should be stored (bars, beats, spectrograms, ssm, ...)
path_annotation_rwcpop = f"{path_parent_of_data}/data/annotations/rwcpop" ## Path of the annotations of RWC Pop. Should be parent folder of "AIST" and/or "MIREX10"
path_entire_rwcpop = "C:/Users/amarmore/Desktop/Audio samples/RWC Pop/Entire RWC" ## Path where are stored wav files of RWC
#path_even_songs_rwcpop = "C:/Users/amarmore/Desktop/Audio samples/RWC Pop/Even songs" ## Path containing only the wav files of songs of odd numbers
#path_odd_songs_rwcpop = "C:/Users/amarmore/Desktop/Audio samples/RWC Pop/Odd songs" ## Path containing only the wav files of songs of even numbers
#path_debug_rwcpop = "C:/Users/amarmore/Desktop/Audio samples/RWC Pop/debug" ## Debug path, containing only two songs, to fix functions quickly
#path_mirdata_rwcpop = "C:/Users/amarmore/Desktop/Audio samples/mirdata/RWC Pop/"

# SALAMI
path_data_persisted_salami = f"{path_parent_of_data}/data/annotations/salami" ## Path where pre-computed data on the SALAMI dataset should be stored (bars, beats, spectrograms, ssm, ...)
path_entire_salami = "C:/Users/amarmore/Desktop/Audio samples/SALAMI" ## Path where are stored wav files of SALAMI (path where it is downloaded by mirdata also)

# Come Together
come_together = "/home/a23marmo/this_folder/The Beatles - Come Together"
path_data_persisted_come_together = "/home/a23marmo/Bureau/data_persisted/cometogether"