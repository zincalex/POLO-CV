#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include "iostream"
#include "BoundingBox.hpp"

class Metrics {
public:
    Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction, const cv::Mat& segmentationColorMask);

    double calculateMeanAveragePrecisionParkingSpaceLocalization() const;

private:
    std::vector<BoundingBox> groundTruth;
    std::vector<BoundingBox> bBoxesPrediction;
    cv::Mat segmentationColorMask;

    int totBoundingBoxes;
};

#endif