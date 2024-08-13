#include "../include/BoundingBox.hpp"

BoundingBox::BoundingBox(const cv::RotatedRect& rect, const unsigned short& number, const bool& occupied) {
    this->center = rect.center;
    this->rect = rect;
    this->number = number;
    this->occupied = occupied;
}

void BoundingBox::updateState() {
    if (!occupied) {
        occupied = true;
    } else {
        occupied = false;
    }
}

cv::Point BoundingBox::getTlCorner() const {
    cv::Point2f vertices[4];
    rect.points(vertices);
    return cv::Point(static_cast<int>(vertices[0].x), static_cast<int>(vertices[0].y));
}

cv::Point BoundingBox::getBrCorner() const {
    cv::Point2f vertices[4];
    rect.points(vertices);
    return cv::Point(static_cast<int>(vertices[2].x), static_cast<int>(vertices[2].y));
}

