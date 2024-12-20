import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="as_seg",
    version="0.1.11",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Package for the segmentation of autosimilarity matrices. This version is related to a stable vesion on PyPi, for installation in MSAF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.imt-atlantique.fr/a23marmo/autosimilarity_segmentation",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.8"
    ],
    license='BSD',
    install_requires=[
        'base_audio',
        'librosa >= 0.10',
        'madmom',# @ git+https://github.com/CPJKU/madmom',
        'matplotlib >= 1.5',
        'mir_eval',
        'mirdata >= 0.3.3',
        'smart_open', # For mirdata, not installed by default, may be fixed in future release
        'numpy >= 1.8.0',
        'pandas',
        'scikit-learn >= 0.17.0',
        'scipy >= 0.13.0',
        'tensorly >= 0.5.1',
        'IPython'
    ]
)
