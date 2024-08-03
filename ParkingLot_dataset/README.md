# Analysis of parking lot occupancy


## Overview
The benchmark dataset consists of 5 different sequences of a surveillance camera monitoring a parking lot, selected from the public available “PKLot” dataset. For each sequence, five frames were selected  corresponding to different times of the day and parking occupancy. 

## Dataset structure

The dataset is organized as the following. A separate folder is provided for each sequence, containing:

- a `frames` folder containing 5 frames extracted from the video surveillance camera;
- a `bounding_boxes` folder containing the bounding box annotations of each image as a XML file;
- a `masks` folder containing the segmentation mask annotations of each image.


## Annotation labels

The benchmark dataset has been annotated according the following tasks: parking space detection and car segmentation. 


#### Parking space detection

For parking space detection, the provided annotation follow the standard defined for the PKLot dataset. For each image, annotations are provided in a XML filec ontaining a list of all the parking spaces.
Every parking space is represented by a set of XML properties, namely: 

- a unique ID number (“id”);
- a value “occupied” describing if the space is occupied (=1) or not (=0)
- a rotated rectangle defined by 5 parameters [x, y, width, height, angle], where (x,y) are the rectangle center coordinates and width and height are the bounding box main dimensions; the 5th parameter is the angle between the main rectangle dimension and the horizontal image dimension (see OpenCV RotatedRect documetation);


#### Car segmentation

For car segmentation, annotations are provided as a grayscale mask where each pixel is assigned the corresponding category ID  (background, parked car, not parked car).Each category is assigned to a unique ID:

0. Background
1. car inside a parking space (parked car)
2. car outside a parking space (not parked car)

Note that segmentation masks can be easily visualized as color images highlighting the different categories by mapping each category ID in a segmentation mask to a RGB color, for example:

0: (128,128,128)
1: (255,0,0)
2: (0,255,0)


## References

- PKLot dataset, https://web.inf.ufpr.br/vri/databases/parking-lot-database/
- RotatedRect, https://docs.opencv.org/4.x/db/dd6/classcv_1_1RotatedRect.html#a6bd95a46f9ab83a4f384a4d4845e6332
- https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/