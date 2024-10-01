## POLO: Parking Occupancy and Lot Observation - Computer Vision
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)


## Overview
Many video surveillance systems rely on computer vision algorithms to identify and track objects of interest in input
videos and generate semantic information. The information obtained from such systems can then be used to monitor parking occupancy over time or
identify possible cars to be fined for incorrect use of the spaces provided. The goal of this project is to develop a computer vision system for parking lot management, capable of detecting
vehicles and available parking spaces by analyzing images of parking lots extracted from surveillance camera frames.

The project has 4 main components: localize all parking spaces, classify all parking spaces according to their occupancy, segment cars into correctly parked or "out of place" and, lastly, represent the current status of the parking lot in a 2D top-view visualization map.
Our implementation has used a selected set of public images taken from the PKLot dataset. To have a deeper understanding of our implementation and ideas, try to have a look at [Report](Report.pdf).

This is a project for the computer vision course given by prof. Ghidoni Stefano in the a.y. 2023/2024

## Minimum Requirements 
- C++ 17.0.0
- Cmake 3.8
- OpenCV 4.8


## Usage
In order to manage all dependencies, CMake has been used. Therefore, to obtain the program executable is necessary to:
```shell 
$ cmake . 
```
Then 
```shell 
$ cmake --build .
```
At this point the main executable is ready at use
```shell 
$ ./main  ParkingLot_dataset/sequence0/frames ParkingLot_dataset/sequenceN
```


## Results
For measuring the system performance we have used mean average precision (mAP) for parking space localization and mean intersection over union (mIoU) for car segmentation.
A more detailed analysis of the results is in the [Report](Report.pdf).

<p align="center">
  <img src="https://github.com/zincalex/POLO-CV/blob/main/ParkingLot_dataset/miou_final.png" />
</p>
<p align="center">
  <img src="https://github.com/zincalex/POLO-CV/blob/main/ParkingLot_dataset/mAP_final.png" />
</p>

