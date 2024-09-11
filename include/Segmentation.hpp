//
// Created by trigger on 9/6/24.
//

#ifndef MAIN_SEGMENTATION_HPP
#define MAIN_SEGMENTATION_HPP

#include "../include/ImageProcessing.hpp"
#include "../include/BoundingBox.hpp"
#include "../include/Graphics.hpp"

#include "filesystem"
#include "vector"
#include <iostream>

#include "string"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/video/background_segm.hpp>


class Segmentation {
public:
    /**
     * @brief Constructor to initialize the class for a Segmentation object.
     *
     * @param emptyFramesDir path to the directory of the image(s) without any car ----> sequence0
     * @param mogTrainigDir  path to the directory of the image(s) without any car used to train the MOG background subtractor ----> mog2_training_sequence
     * @param parkingBBoxes vector of bounding boxes detected in the parking
     * @param imageName  path to the image to load and analyze
     */
    Segmentation(const std::filesystem::path& emptyFramesDir, const std::filesystem::path& mogTrainigDir,const std::vector<BoundingBox>& parkingBBoxes ,const std::string& imageName);

    /**
     * @return mat containing source image with mask applied to it
     */
    cv::Mat getSegmentationResult ();

    /**
     * @return mat containing a color mask to identify classes
     */
    cv::Mat getSegmentationMaskWithClasses ();

    /**
     * @return mat containing a binary mask with segmentation results
     */
    cv::Mat getSegmentationMaskBinary ();

    /**
     * @return mat containing a binary mask with the results of MOG2 in HSV color space for parking space occupation use
     */
    cv::Mat getMOG2HSVmask();

private:
    cv::Mat final_mask;
    cv::Mat final_image;
    cv::Mat final_binary_mask;
    cv::Mat parking_hsv;


    /**
     * @brief Uses the images in the train sequence to train the MOG2 background subtractor.
     *
     * @param backgroundImages set of training images
     * @param color_conversion_code defaults to 0 for BGR images, can accept any cv COLOR_BGR... for other color spaces
     *
     * @return pointer to a trained BackgroundSubtractorMOG2 object
     */
    cv::Ptr<cv::BackgroundSubtractorMOG2> trainBackgroundModel(const std::vector<cv::String>& backgroundImages, const int& color_conversion_code = 0);

    /**
     * @brief Apply the BackgroundSubtractorMOG2 to an image of a busy parking lot, discarding the possible foreground and keeping only the foreground mask.
     *
     * @param mog2 pointer to a trained BackgroundSubtractorMOG2 object
     * @param busy_parking mat containing the image where to apply the background elimination
     *
     * @return mat with a binary mask obtained by the application of MOG2
     */
    cv::Mat getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2>& mog2, cv::Mat& busy_parking);

    /**
     * @brief Calculates the average of the sequence0 images, used in background elimination to improve generalization performance.
     *
     * @param emptyFramesDir path to the sequence0 frames
     *
     * @return mat containing the average of the grayscale images in the folder
     */
    cv::Mat averageEmptyImages(const std::filesystem::path& emptyFramesDir);

    /**
     * @brief Creates the background elimination mask coming from the averaged empty parking images using absolute difference and thresholding.
     *
     * @param empty_parking mat containing an empty parking picture to use as background
     * @param busy_parking mat containing an busy parking picture to use a target
     *
     * @return mat containing a binary mask obtained from background elimination
     */
    cv::Mat backgroundSubtractionMask(const cv::Mat& empty_parking, const cv::Mat& busy_parking);

    /**
     * @brief Eliminates elements smaller than a defined area to eliminate noise.
     *
     * @param input_mask mat containing source mask with noise
     * @param minArea minimum area for the element to be kept
     *
     * @return mat containing mask cleaned from noise
     */
    cv::Mat smallContoursElimination(const cv::Mat& input_mask, const int&minArea);

    /**
     * @brief Generates a binary mask marking the detected parking spaces, used to determine if a segmented car is parked or roaming.
     *
     * @param parkingBBoxes vector of parking space BoundingBoxes
     * @param target empty matrix with the same size of the image to mask
     *
     * @return mat containing a binary mask selecting the parking spaces
     */
    cv::Mat getBBoxMask(const std::vector<BoundingBox>& parkingBBoxes, cv::Mat& target);

    /**
     * @brief Creates a new mask with same ROI as the input one but with colours assigned, giving the segmented elements a class.
     *
     * @param car_fgMask mat with binary mask containing the final segmentation mask
     * @param parking_mask mat with binary mask containing the parking spaces mask
     *
     * @return mat containing a mask with black as background, red as car in parking slot and green as roaming car
     */
    cv::Mat getColorMask(const cv::Mat& car_fgMask, const cv::Mat& parking_mask);

    int dynamicContoursThresh(const cv::Mat& mask_to_filter);
};





#endif //MAIN_SEGMENTATION_HPP
