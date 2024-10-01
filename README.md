## POLO: Parking Observation and Lot Observation - Computer Vision
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)


## Overview
A satellite scans the Earth to acquire images of it. Patches extracted out of these images are used for classification.
The aim is to automatically provide labels describing the represented physical land type or how the land is used.

EUROSAT dataset has been used for this project and there are in total 27,000 labeled 
and georeferenced images, with 13 spectral bands each, consisting of 10 different classes.
The aim of this project is to leverage information extracted from multiple
bands so as to maximize classification performance. 
Two main analysis have been conducted: an ensemble DNN and a PCA dimensionality reduction tecnique applied to each image.

This is a project for the course Deep Learning taken by prof. Loris Nanni in the a.y. 2023/2024

## Requirements 
- Python 3.11.3
- CUDA 12.4.1
- PyTorch 2.2.2 +cu121

## Installation 
Given that the device has CUDA installed, the libraries that have been used
can be downloaded through pip.
- Basic libraries used
```shell
pip install numpy, scikit-learn, h5py, tifffile, matplotlib, tqdm
```
- Pytorch download setup
```shell
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

The dataset is open source and is avaiable via Zenovo [here](https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1).
Then extract the labeled directories, inside the `dataset` directory of this project. Now, you are good to go.

## Usage
To run the main program use the python script `main.py` inside the src folder. Such script accepts various optional parameters: 

```
usage: main.py [-t T] [-d D] [-p P]

optional arguments:
- T   type of analysis: 0 = RGB, 1 = ensemble, 2 = PCA (default = 1)
- D   skip dataset creation: 0 = false, 1 = true (default = 0)
- P   skip PCA version of the dataset: 0 = false, 1 = true (default = 0)
```

The `-t` option specifies the type of analysis to conduct. Only one analysis for run is possible. 

The `-d` option manages whether the images as numpy arrays are saved in the EUROSAT_numpy directory as a .h5 file. 
If the option was activated, the dataset is loaded instead of converting each .tif image to numpy array which saves time for multiple runs of the program.

The `-p` option manages whether the PCA rapresentation of the images is saved in the PCA_dataset directory as a .h5 file (useful for multiple runs of the program, in order to save time).

It is suggested to build each compressed dataset once and after that remember to specify the `-d` and `-p` option to be 1. In this way the next runs will be way faster.

## Results
A more detailed analysis of the results is in the [Report](Report.pdf).
The best results have been achived by using DenseNet161 with final accuracy as 96.9%.

<p align="center">
  <img src="https://github.com/zincalex/Land_Cover_Classification_EUROSAT/blob/main/data/reference_images/densenet161_ENSEMBLE.png" />
</p>
<p align="center">
  <img src="https://github.com/zincalex/Land_Cover_Classification_EUROSAT/blob/main/data/reference_images/PLOT_densenet161_ENSEMBLE.png" />
</p>
