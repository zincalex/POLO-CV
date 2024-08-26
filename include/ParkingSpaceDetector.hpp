#ifndef PARKINGSPACEDETECTION_HPP
#define PARKINGSPACEDETECTION_HPP

#include "../include/BoundingBox.hpp"
#include "../include/ImageProcessing.hpp"

#include <vector>
#include <iostream>
#include <map>
#include <filesystem>
#include <optional>  // Requires C++17 or later
#include <limits> // For std::numeric_limits
#include <algorithm>
#include <functional>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/dnn.hpp>

class ParkingSpaceDetector {
public:
    ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir);

    std::vector<BoundingBox> getBBoxes() { return bBoxes; }

private:
    std::vector<BoundingBox> bBoxes;

    cv::Mat maskCreation(const cv::Mat& inputImg);
    bool isWithinRadius(const cv::Point& center, const cv::Point& point, const double& radius);

    std::vector<cv::RotatedRect> nonMaximaSuppression(const std::vector<cv::RotatedRect>& parkingLotsBoxes, const float& iouThreshold);
    std::vector<cv::RotatedRect> computeAverageRect(const std::vector<std::vector<cv::RotatedRect>>& boundingBoxesNMS);
    std::vector<cv::RotatedRect> rotateBoundingBoxes(const std::vector<std::pair<cv::Point, cv::Rect>>& rects, const float& angle);
};

#endif