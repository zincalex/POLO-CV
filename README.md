## POLO: Parking Occupancy and Lot Observation - Computer Vision
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)


## Overview
Many video surveillance systems rely on computer vision algorithms to identify and track objects of interest in input
videos and generate semantic information. The information obtained from such systems can then be used to monitor parking occupancy over time or
identify possible cars to be fined for incorrect use of the spaces provided. The goal of this project is to develop a computer vision system for parking lot management, capable of detecting
vehicles and available parking spaces by analyzing images of parking lots extracted from surveillance camera frames.

The project has 4 main components: localize all parking spaces, classify all parking spaces according to their occupancy, segment cars into correctly parked or "out of place" and , lastly, represent the current status of the parking lot in a 2D top-view visualization map.
Our implementation has used a selected set of public images taken from the PKLot dataset.

This is a project for the computer vision course given by prof. Ghidoni Stefano in the a.y. 2023/2024

## Requirements 
- Python 3.11.3
- CUDA 12.4.1
- PyTorch 2.2.2 +cu121


## Usage
In order to manage all dependencies, CMake has been used. Therefore, to obtain the program executable is necessary to:

```
first: cmake .
second: cmake --build .

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
