/**
 * @author Alessandro Viespoli 2120824
 */
#include "../include/BoundingBox.hpp"

BoundingBox::BoundingBox(const cv::RotatedRect& rect, const unsigned short& number, const bool& occupied) {
    this->center = rect.center;
    this->rect = rect;
    this->number = number;
    this->occupied = occupied;
}


void BoundingBox::updateState() {
    // occupied = 0 ---> parking space empty
    // occupied = 1 ---> parking space full
    occupied ? occupied = false : occupied = true;
}


cv::Point BoundingBox::getTlCorner() const {
    cv::Point2f vertices[4];
    rect.points(vertices);
    cv::Point topLeft = vertices[0];
    double minSum = topLeft.y;

    // The top left corner has the lowest y coordinate
    for (unsigned int i = 1; i < 4; ++i) {
        double sum = vertices[i].y;
        if (sum < minSum) {
            minSum = sum;
            topLeft = vertices[i];
        }
    }
    return topLeft;
}



