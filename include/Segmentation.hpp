//
// Created by trigger on 9/6/24.
//

#ifndef MAIN_SEGMENTATION_HPP
#define MAIN_SEGMENTATION_HPP

#include "../include/ImageProcessing.hpp"

#include "filesystem"
#include "vector"
#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>


class Segmentation {
public:
    Segmentation(const std::filesystem::path& emptyFramesDir, const std::string& imageName);



private:
    cv::Mat averageEmptyImages(const std::filesystem::path& emptyFramesDir);

    cv::Mat backgroundSubtractionMask(const cv::Mat& empty_parking, const cv::Mat& busy_parking);

    cv::Mat siftMaskEnhancement(const cv::Mat& starting_mask, const cv::Mat& empty_parking, const cv::Mat& masked_busy_parking);

    cv::Mat smallContoursElimination(const cv::Mat& input_mask);

    cv::Mat grabCutMask(const cv::Mat& input_mask, const cv::Mat& input_img);


};



#endif //MAIN_SEGMENTATION_HPP
