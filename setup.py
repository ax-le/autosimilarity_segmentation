import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="as_seg",
    version="0.1.4",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Package for the segmentation of autosimilarity matrices. This version is related to a stable vesion on PyPi, for installation in MSAF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/amarmore/autosimilarity_segmentation",
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
        'librosa >= 0.6.0',
        'madmom @ https://github.com/CPJKU/madmom/tarball/master#egg=madmom-0.17dev',
        'matplotlib >= 1.5',
        'mir_eval',
        'mirdata >= 0.3.3',
        'numpy >= 1.8.0,<1.24',
        'pandas',
        'scikit-learn >= 0.17.0',
        'scipy >= 0.13.0',
        'tensorly >= 0.5.1'
    ]
)
