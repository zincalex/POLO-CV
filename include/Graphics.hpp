#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <algorithm>
#include <vector>
#include <set>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

class Graphics {
public:
    Graphics() = delete;

    static void applyMap(const std::string& imageName, const std::vector<unsigned short>& occupiedParkingSpaces);

private:
    static void getParkingRow(std::vector<cv::RotatedRect>& parkingSlots , int numParking, float angle, int xOffset, float horizontalOffsetAdjustment, int yOffset, float verticalOffsetAdjustment, int parkingWidth, int parkingHeight, int spacing, int extraSpacing = 0, bool isDoubleRow = false, bool isLowerSet = false);

    static void drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects);

    static std::vector<cv::RotatedRect> getBoxes();

    static cv::Mat drawMap(const std::vector<cv::RotatedRect>& parkingSpaces);

    static void fillRotatedRectsWithCar(cv::Mat& image, const std::vector<cv::RotatedRect>& rectangles, const std::vector<unsigned short>& carIndices);

    static void mapOverlay(cv::Mat& src, const cv::Mat& map);
};


#endif
