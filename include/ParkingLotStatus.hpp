#ifndef PARKINGLOTSTATUS_HPP
#define PARKINGLOTSTATUS_HPP

#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <opencv4/opencv2/highgui.hpp>

#include "../include/BoundingBox.hpp"
#include "iostream"

class ParkingLotStatus {
public:
    ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes);


    // metodo per restituire immagine/i con le bbox colorate in base al parcheggio

    // metodo per avere stato di parcheggio con id

    //


private:
    int i = 0;

};

#endif