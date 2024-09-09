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
    Segmentation(const std::filesystem::path& emptyFramesDir, const std::filesystem::path& mogTrainigDir,const std::vector<BoundingBox>& parkingBBoxes ,const std::string& imageName);

    cv::Mat seeSegmentationResult ();

    cv::Mat getSegmentationMask ();

private:
    cv::Mat final_mask;
    cv::Mat final_image;

    cv::Ptr<cv::BackgroundSubtractorMOG2> trainBackgroundModel(const std::vector<cv::String>& backgroundImages);

    cv::Mat getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2>& mog2, cv::Mat& busy_parking);

    cv::Mat averageEmptyImages(const std::filesystem::path& emptyFramesDir);

    cv::Mat backgroundSubtractionMask(const cv::Mat& empty_parking, const cv::Mat& busy_parking);

    cv::Mat smallContoursElimination(const cv::Mat& input_mask, const int&minArea);

    cv::Mat getBBoxMask(const std::vector<BoundingBox>& parkingBBoxes, cv::Mat& target);

    cv::Mat getColorMask(const cv::Mat& car_fgMask, const cv::Mat& parking_mask);
};



#endif //MAIN_SEGMENTATION_HPP
