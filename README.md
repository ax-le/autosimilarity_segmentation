# as_seg: module for computing and segmenting autosimilarity matrices. #

Hello, and welcome on this repository!

This project aims at computing autosimilarity matrices, and segmenting them, which consists of the task of structural segmentation.

The current version contains the CBM algorithm [1], along with a (low-effort) implementation of Foote's novelty algorithm [2].

It can be installed using pip as `pip install as-seg`.

This is a first release, and may contain bug. Comments are welcomed!

## Tutorial notebook ##

A tutorial notebook presenting the most important components of this toolbox is available in the folder "Notebooks".

It is only available if you downloaded the project from git (e.g. https://gitlab.inria.fr/amarmore/autosimilarity_segmentation), and is not available in the pip version (which is in general not accessible easily in the file tree).

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
[1] A. Marmoret, J.E. Cohen, and F. Bimbot, "Convolutive Block-Matching Segmentation Algorithm with Application to Music Structure Analysis", 2022, arXiv preprint arXiv:2210.15356.

[2] J. Foote, "Automatic audio segmentation using a measure of audio novelty," in: 2000 IEEE Int. Conf. Multimedia and Expo. ICME2000. Proc. Latest Advances in the Fast Changing World of Multimedia, vol. 1, IEEE, 2000, pp. 452–455.
