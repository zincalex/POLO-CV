/**
 * @author Francesco Colla 2122543
 */

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

    /**
     * @brief Applies map created from the occupied parking spaces as overlay on the given image
     *
     * @param imageName path of the image to put the map overlay on
     * @param occupiedParkingSpaces vector with the numbers of the busy parking slots used to get map status
     */
    static void applyMap(const std::string& imageName, const std::vector<unsigned short>& occupiedParkingSpaces);

    /**
     * @brief Draws the rotated rectangles on the destination image.
     *
     * @param image mat containing the target image
     * @param rotatedRects vector of rotated rects to draw
     */
    static void drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects);

    /**
     * @brief Takes a mask and applies it to the target matrix.
     *
     * @param target mat containing original image
     * @param mask mat containing mask to apply
     *
     * @return mat containing original image with mask superimposed
     */
    static cv::Mat maskApplication(cv::Mat& target, const cv::Mat& mask);


private:

    /**
     * @brief Creates the rotated rectangles that will create the structure of the map. The methods and parameters allow the creation of
     * different kinds of maps but will be tailored to the layout of the considered parking lot.
     *
     * @param parkingSlots vector of rotated rect, holds the rectangles that compose the map
     * @param numParking number of parking spaces in the current row
     * @param angle angle of the parking spaces in the current row
     * @param xOffset distance of the first row from the left border of the image
     * @param horizontalOffsetAdjustment allows for fine tuning of rotated rect for better fitting - horizontal
     * @param yOffset distance of the first row from the top border of the image
     * @param verticalOffsetAdjustment allows for fine tuning of rotated rect for better fitting - vertical
     * @param parkingWidth size of the individual parking slot - width TODO:check if should be hardcoded
     * @param parkingHeight size of the individual parking slot - width
     * @param spacing distance between parking spacing
     * @param extraSpacing additional offset to help handling more extreme angles
     * @param isDoubleRow true if considering a set of parking slots with two rows (middle and bottom)
     * @param isLowerSet true if considering the lower row of parking spaces in a set with two rows
     */
    static void getParkingRow(std::vector<cv::RotatedRect>& parkingSlots , int numParking, float angle, int xOffset, float horizontalOffsetAdjustment, int yOffset, float verticalOffsetAdjustment, int parkingWidth, int parkingHeight, int spacing, int extraSpacing = 0, bool isDoubleRow = false, bool isLowerSet = false);

    static std::vector<cv::RotatedRect> getBoxes();

    /**
     * @brief Creates the map given the rotated rectangles built for this parking lot.
     *
     * @param parkingSpaces vector containing the rectangles representing the parking slots
     * @return
     */
    static cv::Mat drawMap(const std::vector<cv::RotatedRect>& parkingSpaces);

    /**
     * @brief Fills the parking slots with the correct colors to represent free or busy slots. Upper portion is ignored by design and blacked.
     *
     * @param empty_map mat containing the already drawn map with rectangles to be filled
     * @param rectangles vector containing the rectangles to fill
     * @param carIndices vector containing the indexes of the busy parking slots
     */
    static void fillRotatedRectsWithCar(cv::Mat& empty_map, const std::vector<cv::RotatedRect>& rectangles, const std::vector<unsigned short>& carIndices);

    /**
     * @brief Support function called to overlay the map.
     *
     * @param src mat containing target image
     * @param map mat containing the already built map
     */
    static void mapOverlay(cv::Mat& src, const cv::Mat& map);
};


#endif
