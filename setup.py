import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="as_seg",
    version="0.1.1waspaa23",
    author="Marmoret Axel",
    author_email="axle_le@protonmail.com",
    description="Package for the segmentation of autosimilarity matrices. This version is related to a publication in WASPAA 2023.",
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
        'librosa == 0.8.1',
        'madmom == 0.16.1',
        'matplotlib',
        'mir_eval == 0.6',
        'mirdata == 0.3.3',
        'numpy == 1.22.4',
        'pandas',
        'scipy == 1.5.4',
        'scikit-learn',
        'soundfile',
        'tensorly == 0.5.1'
    ],
    python_requires='>=3.7.*',
)
