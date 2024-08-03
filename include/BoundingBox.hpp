#ifndef BOUNDINGBOX_HPP
#define BOUNDINGBOX_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

class BoundingBox {
public:
    BoundingBox(const cv::Mat& input);

    cv::Mat getImg() {return img;}

private:
    cv::Mat img;
};



#endif