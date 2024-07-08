# as_seg: module for computing and segmenting autosimilarity matrices. #

Hello, and welcome on this repository!

This project aims at computing autosimilarity matrices, and segmenting them, which consists of the task of structural segmentation.

The current version contains the CBM algorithm [1], along with an implementation of Foote's novelty algorithm [2] based on the MSAF toolbox [3].

It can be installed using pip as `pip install as-seg`.

This is a first release, and may contain bug. Comments are welcomed!

## Tutorial notebook ##

A tutorial notebook presenting the most important components of this toolbox is available in the folder "Notebooks".

## Experimental notebook ##

A Tutorial notebook is presented in the "Notebooks" folder. In older version of the code, you may find Notebooks presenting experiments associated with publications.

## Data ##

Should be obtained from Zenodo: https://zenodo.org/records/10168387. DOI: 10.5281/zenodo.10168386.

## Software version ##

This code was developed with Python 3.8.5, and some external libraries detailed in dependencies.txt. They should be installed automatically if this project is downloaded using pip.

## How to cite ##

You should cite the package `as_seg`, available on HAL (https://hal.archives-ouvertes.fr/hal-03797507).

Here are two styles of citations:

As a bibtex format, this should be cited as: @softwareversion{marmoret2022as_seg, title={as\_seg: module for computing and segmenting autosimilarity matrices}, author={Marmoret, Axel and Cohen, J{\'e}r{\'e}my and Bimbot, Fr{\'e}d{\'e}ric}, LICENSE = {BSD 3-Clause ''New'' or ''Revised'' License}, year={2022}}

In the IEEE style, this should be cited as: A. Marmoret, J.E. Cohen, and F. Bimbot, "as_seg: module for computing and segmenting autosimilarity matrices," 2022, url: https://gitlab.inria.fr/amarmore/autosimilarity_segmentation.

## Credits ##

Code was created by Axel Marmoret (<axel.marmoret@imt-atlantique.fr>), and strongly supported by Jeremy E. Cohen (<jeremy.cohen@cnrs.fr>).

The technique in itself was also developed by Frédéric Bimbot (<bimbot@irisa.fr>).

## References ##
[1] A. Marmoret, J.E. Cohen, F. Bimbot. Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm. Transactions of the International Society for Music Information Retrieval (TISMIR), 2023, 6 (1), pp.167-185. ⟨10.5334/tismir.167⟩. ⟨hal-04323556⟩, https://hal.science/hal-04323556.

[2] J. Foote, "Automatic audio segmentation using a measure of audio novelty," in: 2000 IEEE Int. Conf. Multimedia and Expo. ICME2000. Proc. Latest Advances in the Fast Changing World of Multimedia, vol. 1, IEEE, 2000, pp. 452–455.

[3] Nieto, O., Bello, J. P., Systematic Exploration Of Computational Music Structure Research. Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR). New York City, NY, USA, 2016.

[4] Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., & Widmer, G. (2016, October). Madmom: A new python audio and music signal processing library. In Proceedings of the 24th ACM international conference on Multimedia (pp. 1174-1178).
