#ifndef PARKINGLOTSTATUS_HPP
#define PARKINGLOTSTATUS_HPP

#include "../include/BoundingBox.hpp"
#include "../include/ImageProcessing.hpp"

#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

class ParkingLotStatus {
public:
    ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes);

    void drawParkingLotStatus();
    // metodo per avere stato di parcheggio con id

    cv::Mat getStatusImage() const { return parkingImage; }
    std::vector<BoundingBox> getStatusPredictions() const { return bBoxes; }

private:
    cv::Mat parkingImage;
    std::vector<BoundingBox> bBoxes;

    bool isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const;
    cv::Mat createMaskBlackishColors(const cv::Mat& image) const;
};


#endif