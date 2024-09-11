/**
 * @author Francesco Colla 2122543
 */
#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <vector>

#include <opencv2/imgproc.hpp>


class Graphics {
public:

    Graphics() = delete;   // Prevent instantiation, class used as library

    /**
     * @brief Applies a map created from the occupied parking spaces as overlay on the given image
     *
     * @param src                     image to put the map overlay on
     * @param occupiedParkingSpaces   vector with the numbers of the busy parking slots
     */
    static void applyMap(cv::Mat& src, const std::vector<unsigned short>& occupiedParkingSpaces);

    /**
     * @brief Draws the rotated rectangles on the given image.
     *
     * @param image          target image
     * @param rotatedRects   vector of rotated rects to draw
     */
    static void drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects);

    /**
     * @brief Takes a mask and applies it to the target image.
     *
     * @param target    original image
     * @param mask      mask to apply
     *
     * @return a cv::Mat containing original image with mask superimposed
     */
    static cv::Mat maskApplication(cv::Mat& target, const cv::Mat& mask);


private:

    /**
     * @brief Add a set of rotated rectangles which are part of a single row of the parking lot. Changing the parameters allow the creation of
     *        different kinds of rows that can be tailored to the layout of the considered parking lot.
     *
     * @param parkingSlots                   vector of rotated rect, holds the rectangles that compose the map
     * @param numParking                     number of parking spaces in the current row
     * @param angle                          angle of the parking spaces in the current row
     * @param xOffset                        distance of the first row from the left border of the image
     * @param horizontalOffsetAdjustment     allows for fine tuning of rotated rect for better fitting - horizontal
     * @param yOffset                        distance of the first row from the top border of the image
     * @param verticalOffsetAdjustment       allows for fine tuning of rotated rect for better fitting - vertical
     * @param parkingWidth                   size of the individual parking slot - width
     * @param parkingHeight                  size of the individual parking slot - height
     * @param spacing                        distance between parking spaces
     * @param extraSpacing                   additional offset to help handling more extreme angles
     * @param isDoubleRow                    true if the row wer are considering is in a double row configuration. False if the row is isolated
     * @param isLowerSet                     true if considering the lower row of parking spaces in a set with two rows
     */
    static void createParkingRow(std::vector<cv::RotatedRect>& parkingSlots , int numParking, float angle, int xOffset, float horizontalOffsetAdjustment, int yOffset, float verticalOffsetAdjustment, int parkingWidth, int parkingHeight, int spacing, int extraSpacing = 0, bool isDoubleRow = false, bool isLowerSet = false);

    /**
     * @brief Generate and retrive the rotated rects in order to build the overlay map ---> Parking lot specific.
     *
     * @return a vector with all the parking spaces represented for the map configuration
     */
    static std::vector<cv::RotatedRect> getBoxesForMap();

    /**
     * @brief Creates the map given the rotated rectangles built for this parking lot.
     *
     * @param parkingSpaces vector containing the rectangles representing the parking slots
     *
     * @return the map
     */
    static cv::Mat drawMap(const std::vector<cv::RotatedRect>& parkingSpaces);

    /**
     * @brief Fills the parking slots with the correct colors to represent free or busy slots. Upper portion is ignored by design.
     *
     * @param empty_map    already drawn map with rectangles to be filled
     * @param rectangles   vector containing the rectangles to fill
     * @param carIndices   vector containing the indexes of the busy parking slots
     */
    static void fillRotatedRectsWithCar(cv::Mat& empty_map, const std::vector<cv::RotatedRect>& rectangles, const std::vector<unsigned short>& carIndices);

    /**
     * @brief Overlay a map on the given image
     *
     * @param src   cv::Mat containing target image
     * @param map   cv::Mat containing the already built map
     */
    static void mapOverlay(cv::Mat& src, const cv::Mat& map);
};


#endif
