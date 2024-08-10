#ifndef PARKINGSPACEDETECTION_HPP
#define PARKINGSPACEDETECTION_HPP

#include <vector>
#include <iostream>
#include <map>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

class ParkingSpaceDetector {
public:
    ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir);

    cv::Mat getImg() {return img;}

private:
    cv::Mat img;

    cv::Mat createROI(const cv::Mat& input);
    cv::Mat gamma_correction(const cv::Mat& input, const double& gamma);
    cv::Mat niBlack_thresholding(const cv::Mat& input, const int& blockSize, const double& k);
    cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold);
    cv::Mat minFilter(const cv::Mat& src, const int& kernel_size);
    bool isWithinRadius(const std::pair<int, int>& center, const std::pair<int, int>& point, const double& radius);
};

#endif