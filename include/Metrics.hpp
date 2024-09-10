/**
 * @author Alessandro Viespoli 2120824
 */

#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <opencv2/imgproc.hpp>

#include "ImageProcessing.hpp"
#include "BoundingBox.hpp"

class Metrics {
public:

    /**
     * @brief Constructor to initialize the class for calculating the project metrics.
     *
     * @param groundTruth             vector with the actual composition and status of the parking space
     * @param bBoxesPrediction        vector with the prediction of the composition and status of the parking space
     * @param trueSegmentationMask    mask that represent the true semantic segmentation of the parking lot
     * @param segmentationColorMask   mask that represent the predicted semantic segmentation of the parking lot
     */
    Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction,
            const cv::Mat& trueSegmentationMask, const cv::Mat& segmentationColorMask);

    /**
     * @brief Calculates the Mean Average Precision (mAP) for parking space localization at IoU threshold at 0.5 for
     *        two classes: car, no car. The method used for calculating the AP is the 11-point interpolation method.
     *
     * @return the mAP value
     */
    double calculateMeanAveragePrecisionParkingSpaceLocalization() const;

    /**
     * @brief Calculates the Mean Intersection over Union (IoU) for segmentation task. The classes are three: background (0),
     *        car inside parking space (1), car outside parking space (2).
     *
     * @return the mIoU value
     */
    double calculateMeanIntersectionOverUnionSegmentation()  const;


private:
    std::vector<BoundingBox> groundTruth;
    std::vector<BoundingBox> bBoxesPrediction;
    cv::Mat trueSegmentationMask;
    cv::Mat segmentationColorMask;


    /**
     * @brief Calculates the Intersection over Union (IoU) between two rotated rotated rects coming from the BoundingBox object.
     *
     * @param rect1     first rotated rectangle
     * @param rect2     second rotated rectangle
     *
     * @return the IoU value
     */
    double calculateIoUObjectDetection(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) const;

    /**
     * @brief Calculates the Intersection over Union (IoU) for semantic segmentation for a specific class.
     *
     * @param groundTruthMask   ground truth segmentation mask
     * @param predictedMask     predicted segmentation mask
     * @param classId           class ID for which the IoU is being calculated.
     *                             - 0 -> background
     *                             - 1 -> car inside parking space
     *                             - 2 -> car outside parking space
     *
     * @return the IoU value, otherwise -1 if the class is not present in both the ground truth and predicted mask
     */
    double calculateIoUSegmentation(const cv::Mat& groundTruthMask, const cv::Mat& predictedMask, const unsigned int& classId) const;
};

#endif