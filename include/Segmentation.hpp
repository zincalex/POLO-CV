/**
 * @author Francesco Colla 2122543
 */
#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <filesystem>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include "BoundingBox.hpp"


class Segmentation {
public:

    /**
     * @brief Constructor to initialize a Segmentation object.
     *
     * @param mogTrainigDir    directory of the images without any car used to train the MOG background subtractor ----> mog2_training_sequence
     * @param parkingBBoxes    vector of bounding boxes detected in the parking
     * @param imageName        path to the image to load and analyze
     */
    Segmentation(const std::filesystem::path& mogTrainingDir, const std::vector<BoundingBox>& parkingBBoxes ,const std::string& imageName);

    /**
     * @return source image with mask applied to it
     */
    cv::Mat getSegmentationResult () const { return final_image; }

    /**
     * @return color mask to identify classes
     */
    cv::Mat getSegmentationMaskWithClasses () const { return final_mask; }

    /**
     * @return binary mask with segmentation results
     */
    cv::Mat getSegmentationMaskBinary () const { return final_binary_mask; }


private:
    cv::Mat final_mask;
    cv::Mat final_image;
    cv::Mat final_binary_mask;
    cv::Mat mog2MaskLab;
    cv::Mat mog2MaskBGR;

    /**
     * @brief Get the busy parking image, confronts it with all images in training set and finds the one with smallest difference,
     *        using it to create a binary mask that identifies only the changing parts of the image.
     *
     * @param emptyFramesDir   directory of the training dataset
     * @param parkingLotImg    containing the image to process
     *
     * @return binary mask of the changing elements
     */
    cv::Mat backgroundSubtraction(const std::filesystem::path &mogTrainingDir, const cv::Mat &parkingLotImg) const ;

    /**
     * @brief Eliminate elements smaller than a defined area in order to eliminate noise.
     *
     * @param inputMask   image containing source mask with noise
     * @param minArea      minimum area for the element to be kept
     *
     * @return mask cleaned from noise
     */
    cv::Mat contoursElimination(const cv::Mat& inputMask, const int&minArea) const;

    /**
     * @brief Use the images in the train sequence to train the MOG2 background subtractor.
     *
     * @param backgroundImages         set of training images
     * @param colorConversionCode    default to 0 for BGR images, can accept any cv COLOR_BGR... for other color spaces
     *
     * @return pointer to a trained BackgroundSubtractorMOG2 object
     */
    cv::Ptr<cv::BackgroundSubtractorMOG2> trainBackgroundModel(const std::vector<cv::String>& backgroundImages, const int& colorConversionCode = 0) const;

    /**
     * @brief Apply the BackgroundSubtractorMOG2 to an image of a busy parking lot, discarding the possible foreground and keeping only the foreground mask.
     *
     * @param mog2             pointer to a trained BackgroundSubtractorMOG2 object
     * @param parkingLotImage  image where to apply the background elimination
     *
     * @return a binary mask obtained by the application of MOG2
     */
    cv::Mat getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2>& mog2, const cv::Mat& parkingLotImage) const;

    /**
     * @brief Generate a binary mask marking the detected parking spaces, used to determine if a segmented car is parked or roaming.
     *
     * @param parkingBBoxes  vector of parking space BoundingBoxes
     * @param target         empty matrix with the same size of the image to mask
     *
     * @return  a binary mask selecting the parking spaces
     */
    cv::Mat getBBoxMask(const std::vector<BoundingBox>& parkingBBoxes, const cv::Mat& target) const;

    /**
     * @brief Creates a new mask with same ROI as the input one but with colours assigned, giving the segmented elements a class.
     *
     * @param segmentationMask     binary mask containing the final segmentation mask
     * @param parkingSpacesMask   binary mask containing the parking spaces mask
     *
     * @return a mask with black as background, red as car in parking slot and green as roaming car
     */
    cv::Mat getColorMask(const cv::Mat& segmentationMask, const cv::Mat& parkingSpacesMask) const;
};

#endif