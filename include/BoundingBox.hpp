#ifndef BOUNDINGBOXES_HPP
#define BOUNDINGBOXES_HPP

#include <opencv2/imgproc.hpp>

class BoundingBox {
public:
    BoundingBox(const cv::Point& center, const cv::Rect& rect, const unsigned short& number, const bool& occupied = false);

    void updateState();

    cv::Point getTlCorner() {return rect.tl();}
    cv::Point getBrCorner() {return rect.br();}
    cv::Point getCenter() {return center;}
    unsigned short getHeight() {return rect.height;}
    unsigned short getWidth() {return rect.width;}
    unsigned short getNumber() {return number;}
    //unsigned short getAngle() {return angle;}
    bool isOccupied() {return occupied;}

private:

    cv::Point center;
    cv::Rect rect;
    unsigned short number;
    //unsigned short angle;
    bool occupied;
};

#endif