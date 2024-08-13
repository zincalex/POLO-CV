#ifndef BOUNDINGBOXES_HPP
#define BOUNDINGBOXES_HPP

#include <opencv2/imgproc.hpp>

class BoundingBox {
public:
    BoundingBox(const cv::RotatedRect& rect, const unsigned short& number, const bool& occupied = false);

    void updateState();

    cv::Point getTlCorner() const;
    cv::Point getBrCorner() const;

    float getAngle() const           { return rect.angle; }
    bool isOccupied() const          { return occupied; }
    cv::Point getCenter() const      { return center; }
    unsigned short getNumber() const { return number; }
    unsigned short getHeight() const { return static_cast<unsigned short>(rect.size.height); }
    unsigned short getWidth()  const { return static_cast<unsigned short>(rect.size.height); }


private:
    cv::Point center;
    cv::RotatedRect rect;
    unsigned short number;
    bool occupied;
};

#endif