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


protected:


private:
    std::vector<BoundingBox> bBoxes;

    double calculateLineLength(const cv::Vec4i& line) const;
    double calculatePointsDistance (const cv::Point& pt1, const cv::Point& pt2) const;
    double calculateLineAngle(const cv::Vec4i& line) const;
    bool areAnglesSimilar(const double& angle1, const double& angle2, const double& angleThreshold) const;
    bool isInRange(const double& angle, const std::pair<double, double>& range) const;
    bool isWithinRadius(const cv::Point& center, const cv::Point& point, const double& radius) const;

    cv::Vec4i standardizeLine(const cv::Vec4i& line) const;

    std::vector<cv::Vec4i> filterLines(std::vector<cv::Vec4i>& lines, const cv::Mat& referenceImage, const std::vector<std::pair<double,
                                       double>>& parkingSpaceLinesAngles, const double& proximityThreshold1, const double& proximityThreshold2,
                                       const double& minLength, const double& angleThreshold, const double& whitenessThreshold) const;

    std::vector<std::pair<cv::Vec4i, cv::Vec4i>> matchLines(const std::vector<cv::Vec4i>& linesSupreme, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                                            const double& startEndDistanceThreshold, const double& endStartDistanceThreshold, const double& angleThreshold,
                                                            const double& deltaXThreshold, const double& deltaYThreshold) const;

    std::vector<cv::RotatedRect> linesToRotatedRect(const std::vector<std::pair<cv::Vec4i, cv::Vec4i>>& matchedLines) const ;

    void removeOutliers(std::vector<cv::RotatedRect>& rotatedRects, std::vector<std::pair<double, double>>& parkingSpaceLinesAngles, const cv::Size& imgSize,
                        const int& margin, const std::vector<double>& aspectRatioThresholds) const;






    std::vector<cv::RotatedRect> nonMaximaSuppression(const std::vector<cv::RotatedRect>& parkingLotsBoxes, const float& iouThreshold);
    std::vector<cv::RotatedRect> computeAverageRect(const std::vector<std::vector<cv::RotatedRect>>& boundingBoxesNMS);
    std::vector<cv::RotatedRect> rotateBoundingBoxes(const std::vector<std::pair<cv::Point, cv::Rect>>& rects, const float& angle);

};

#endif