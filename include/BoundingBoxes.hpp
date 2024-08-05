#ifndef BOUNDINGBOXES_HPP
#define BOUNDINGBOXES_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

class BoundingBoxes {
public:
    BoundingBoxes(const cv::Mat& input);

    cv::Mat getImg() {return img;}

private:
    cv::Mat img;

    cv::Mat createROI(const cv::Mat& input);
};

#endif