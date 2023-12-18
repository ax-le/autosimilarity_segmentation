# as_seg: module for computing and segmenting autosimilarity matrices. #

Hello, and welcome on this repository!

This project aims at computing autosimilarity matrices, and segmenting them, which consists of the task of structural segmentation.

The current version contains the CBM algorithm [1], along with a (low-effort) implementation of Foote's novelty algorithm [2].

It can be installed using pip as `pip install as-seg`.

This is a first release, and may contain bug. Comments are welcomed!

## Tutorial notebook ##

A tutorial notebook presenting the most important components of this toolbox is available in the folder "Notebooks".

## Experimental notebook ##

Experimental notebooks are available in the folder "Notebooks". They present the code used to compute the main experiments of the paper, in order to improve the reproducibility. Please tell me if any problem would appear when trying to launch them.

## Data ##

Some data is available with the code, in the folder "data". This includes the bar estimates, obtained with the madmom toolbox [3], the Barwise TF matrices, which are the barwise pre-processed versions of the spectrograms we use to estimate boundaries, and the estimated boundaries obtained with the CBM algorithm in the different conditions.

## Software version ##

This code was developed with Python 3.8.5, and some external libraries detailed in dependencies.txt. They should be installed automatically if this project is downloaded using pip.

## How to cite ##

You should cite the package `as_seg`, available on HAL (https://hal.archives-ouvertes.fr/hal-03797507).

Here are two styles of citations:

As a bibtex format, this should be cited as: @softwareversion{marmoret2022as_seg, title={as\_seg: module for computing and segmenting autosimilarity matrices}, author={Marmoret, Axel and Cohen, J{\'e}r{\'e}my and Bimbot, Fr{\'e}d{\'e}ric}, URL={https://gitlab.inria.fr/amarmore/autosimilarity_segmentation}, LICENSE = {BSD 3-Clause ''New'' or ''Revised'' License}, year={2022}}

In the IEEE style, this should be cited as: A. Marmoret, J.E. Cohen, and F. Bimbot, "as_seg: module for computing and segmenting autosimilarity matrices," 2022, url: https://gitlab.inria.fr/amarmore/autosimilarity_segmentation.

## Credits ##

Code was created by Axel Marmoret (<axel.marmoret@gmail.com>), and strongly supported by Jeremy E. Cohen (<jeremy.cohen@cnrs.fr>).

The technique in itself was also developed by Frédéric Bimbot (<bimbot@irisa.fr>).

## References ##
[1] A. Marmoret, J.E. Cohen, and F. Bimbot, "Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis", 2023, to be published at WASPAA 2023.

[2] J. Foote, "Automatic audio segmentation using a measure of audio novelty," in: 2000 IEEE Int. Conf. Multimedia and Expo. ICME2000. Proc. Latest Advances in the Fast Changing World of Multimedia, vol. 1, IEEE, 2000, pp. 452–455.

[3] Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., & Widmer, G. (2016, October). Madmom: A new python audio and music signal processing library. In Proceedings of the 24th ACM international conference on Multimedia (pp. 1174-1178).
