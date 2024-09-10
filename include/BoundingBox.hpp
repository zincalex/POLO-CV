#ifndef BOUNDINGBOX_HPP
#define BOUNDINGBOX_HPP

#include <opencv2/imgproc.hpp>

class BoundingBox {
public:

    /**
     * @brief Constructor to initialize a Bounding Box object.
     *
     * @param rect        cv::RotatedRect object representing the rotated bounding box
     * @param number      parking space number
     * @param occupied    indicator state whether the parking space is occupied or not. If not specified empty by default
     */
    BoundingBox(const cv::RotatedRect& rect, const unsigned short& number, const bool& occupied = false);

    /**
     * @brief Update the state of the parking space. If empty becomes occupied, and viceversa.
     */
    void updateState();

    /**
     * @return the state of the bounding box (parking space)
     */
    bool isOccupied() const                { return occupied; }

    /**
     * @return the top left corner of the bounding box
     */
    cv::Point getTlCorner() const;

    /**
     * @return the center of the bounding box
     */
    cv::Point getCenter() const            { return center; }

    /**
     * @return the size of bounding box as a cv::Size type
     */
    cv::Size getSize() const               { return rect.size; }

    /**
     * @return the rotated rect that represent the bounding box
     */
    cv::RotatedRect getRotatedRect() const { return rect; }

    /**
     * @return the angle of the bounding box
     */
    double getAngle() const                 { return rect.angle; }

    /**
     * @return the parking space number
     */
    unsigned short getNumber() const       { return number; }


private:
    cv::Point center;
    cv::RotatedRect rect;
    unsigned short number;
    bool occupied;
};

#endif