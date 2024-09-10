/**
 * @author Alessandro Viespoli 2120824
 */

#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include <opencv2/imgproc.hpp>

#include "BoundingBox.hpp"

class ImageProcessing {
public:
    ImageProcessing() = delete;  // Prevent instantiation


    static cv::Mat optionalAreaROI(const cv::Size& imgSize);
    static cv::Mat createRectsMask(const std::vector<cv::RotatedRect>& rotatedRects, const cv::Size& imgSize);
    static cv::Mat createROI(const cv::Mat& image, const BoundingBox& bBox);

    static cv::Mat gamma_correction(const cv::Mat& input, const double& gamma);
    static cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold);
    static cv::Mat createMaskDarkColors(const cv::Mat& image);

    static cv::Mat convertColorMaskToGray(const cv::Mat& segmentationColorMask);
};

#endif