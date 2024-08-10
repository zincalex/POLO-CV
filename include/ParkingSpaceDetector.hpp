#ifndef PARKINGSPACEDETECTION_HPP
#define PARKINGSPACEDETECTION_HPP

#include <vector>
#include <iostream>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

class ParkingSpaceDetection {
public:
    ParkingSpaceDetection(const std::filesystem::path& emptyFramesDir);

    cv::Mat getImg() {return img;}

private:
    cv::Mat img;

    cv::Mat createROI(const cv::Mat& input);

    static cv::Mat gamma_correction(const cv::Mat& input, const double& gamma);
    static cv::Mat niBlack_thresholding(const cv::Mat& input, const int& blockSize, const double& k);
    static cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold);
};

#endif