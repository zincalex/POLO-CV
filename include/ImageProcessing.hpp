/**
 * @author Alessandro Viespoli 2120824
 */
#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include <vector>
#include <opencv2/imgproc.hpp>

#include "BoundingBox.hpp"


class ImageProcessing {
public:

    ImageProcessing() = delete;  // Prevent instantiation, class used as library

    /**
     * @brief Creates a region of interest mask for the optional set of parking spaces in the reference image.
     *
     * @param imgSize size of the image
     *
     * @return a binary mask
     */
    static cv::Mat optionalAreaROI(const cv::Size& imgSize);

    /**
     * @brief Extracts a region of interest (ROI) from the image based on the bounding box.
     *
     * @param image reference image from where to extract the ROI
     * @param bBox  position and dimension of the location to extract
     *
     * @return ROI extracted from the input image
     */
    static cv::Mat createROI(const cv::Mat& image, const BoundingBox& bBox);

    /**
     * @brief Creates a binary mask from a set of rotated rectangles.
     *
     * @param rotatedRects
     * @param imgSize        size of mask
     *
     * @return a binary mask
     */
    static cv::Mat createRectsMask(const std::vector<cv::RotatedRect>& rotatedRects, const cv::Size& imgSize);

    /**
     * @brief Create a binary mask for the darker area of the image.
     *
     * @param image
     *
     * @return a binary mask
     */
    static cv::Mat createMaskDarkColors(const cv::Mat& image);

    /**
     * @brief Converts a color segmentation mask to a grayscale mask with class labels.
     *
     * @param segmentationColorMask  input color segmentation mask in BGR format
     *
     * @return a grayscale mask where pixel values represent class IDs
     *             - 0: Nothing (background)
     *             - 1: Car inside parking space
     *             - 2: Car outside parking space
     */
    static cv::Mat convertColorMaskToGray(const cv::Mat& segmentationColorMask);

    /**
     * @brief Apply a gamma correction to the given image
     *
     * @param input  image to correct
     * @param gamma  correction factor. A value of gamma < 1 will lighten the image, while a value
     *               of gamma > 1 will darken the image.
     *
     * @return corrected image
     */
    static cv::Mat gamma_correction(const cv::Mat& input, const double& gamma);

    /**
     * @brief Convert the BGR image to HSV and apply a saturation thresholding
     *
     * @param input           bgr image to threshold
     * @param satThreshold    saturation value. Values above this will be set to 255, otherwise to 0
     *
     * @return a binary mask
     */
    static cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold);
};

#endif