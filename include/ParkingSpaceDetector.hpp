#ifndef PARKINGSPACEDETECTION_HPP
#define PARKINGSPACEDETECTION_HPP

#include <vector>
#include <iostream>
#include <map>
#include <filesystem>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/dnn.hpp>

#include "../include/BoundingBox.hpp"

class ParkingSpaceDetector {
public:
    ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir);

    std::vector<BoundingBox> getBBoxes() { return bBoxes; }

private:
    std::vector<BoundingBox> bBoxes;

    cv::Mat maskCreation(const cv::Mat& inputImg);
    cv::Mat createROI(const cv::Mat& input);
    cv::Mat gamma_correction(const cv::Mat& input, const double& gamma);
    cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold);
    cv::Mat minFilter(const cv::Mat& input, const int& kernel_size);
    bool isWithinRadius(const std::pair<int, int>& center, const std::pair<int, int>& point, const double& radius);

    std::map<std::pair<int, int>, cv::Rect> nonMaximaSuppression(const std::map<std::pair<int, int>, cv::Rect>& parkingLotsBoxes, const float& iouThreshold);
    std::vector<std::pair<cv::Point, cv::Rect>> computeAverageRect(const std::vector<std::map<std::pair<int, int>, cv::Rect>>& boundingBoxesNMS);
    std::vector<cv::RotatedRect> rotateBoundingBoxes(const std::vector<std::pair<cv::Point, cv::Rect>>& rects, const float& angle);
};

#endif