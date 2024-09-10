#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "iostream"

#include "ImageProcessing.hpp"
#include "BoundingBox.hpp"

class Metrics {
public:
    Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction,
            const cv::Mat& trueSegmentationMask, const cv::Mat& segmentationColorMask);

    double calculateMeanAveragePrecisionParkingSpaceLocalization() const;
    double calculateMeanIntersectionOverUnionSegmentation()  const;

private:
    std::vector<BoundingBox> groundTruth;
    std::vector<BoundingBox> bBoxesPrediction;
    cv::Mat trueSegmentationMask;
    cv::Mat segmentationColorMask;

    int totBoundingBoxes;
};

#endif