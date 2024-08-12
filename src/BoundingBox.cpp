#include "../include/BoundingBox.hpp"


BoundingBox::BoundingBox(const cv::Point& center, const cv::Rect& rect, const unsigned short& number, const bool& occupied) {
    this->center = center;
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