#ifndef PARKINGLOTSTATUS_HPP
#define PARKINGLOTSTATUS_HPP

#include "BoundingBox.hpp"
#include "ImageProcessing.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


class ParkingLotStatus {
public:
    ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes);

    cv::Mat seeParkingLotStatus();

    std::vector<unsigned short> getOccupiedParkingSpaces() const;
    std::vector<BoundingBox> getStatusPredictions() const { return bBoxes; }

private:
    cv::Mat parkingImage;
    std::vector<BoundingBox> bBoxes;

    bool isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const;
};


#endif