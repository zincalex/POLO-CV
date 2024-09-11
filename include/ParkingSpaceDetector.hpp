/**
 * @author Alessandro Viespoli 2120824
 */
#ifndef PARKINGSPACEDETECTION_HPP
#define PARKINGSPACEDETECTION_HPP

#include <vector>
#include <filesystem>
#include <opencv2/imgproc.hpp>

#include "BoundingBox.hpp"


class ParkingSpaceDetector {
public:

    /**
     * @brief Constructor to initialize the class for parking Space detector.
     *        The method calculate the bounding boxes for each empty image inside the given directory, then it
     *        proceds to aggregate the bounding boxes by first checking which represents the same parking space
     *        and then make and avereage. At that point prospective adjustment are made, for later classification, and
     *        number labels are assigned to each parking space.
     *
     * @param emptyFramesDir path to the directory of the image(s) without any car ----> sequence0
     *
     * @throw std::invalid_argument if there are no images inside the given directory
     */
    ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir);

    /**
     * @return vector of BoundingBox objects that represent the parking spaces
     */
    std::vector<BoundingBox> getBBoxes() { return bBoxes; }


private:
    std::vector<BoundingBox> bBoxes;  // BoundingBox objects that represent the parking spaces

    /**
     * @brief Calculate the length of a line.
     */
    double calculateLineLength(const cv::Vec4i& line) const;

    /**
     * @brief Calculate the angle of a line.
     */
    double calculateLineAngle(const cv::Vec4i& line) const;

    /**
     * @brief Calculate the distance among two given points.
     */
    double calculatePointsDistance (const cv::Point& pt1, const cv::Point& pt2) const;

    /**
     * @brief Given two angles, state whether their absolute difference is less than a threshold.
     *
     * @param angleThreshold threshold considered
     *
     * @return true if the angles are similar, false otherwise
     */
    bool areAnglesSimilar(const double& angle1, const double& angle2, const double& angleThreshold) const;

    /**
     * @brief Given an angle, state whether its value is between two values.
     *
     * @param range pair of angles that represent the region considered
     *
     * @return true if the angle is within range, false otherwise
     */
    bool isInRange(const double& angle, const std::pair<double, double>& range) const;

    /**
     * @brief Check if two points are within a radius.
     *
     * @return true if the distance among the two points is less or equal than the radius, false otherwise
     */
    bool isWithinRadius(const cv::Point& center, const cv::Point& point, const double& radius) const;

    /**
     * @brief Check whether the top left corner of the first bounding box is inside the area of the second bounding box.
     *
     * @param bbox1 BoundingBox object of which consider the top left corner
     * @param bbox2 BoundingBox object of which consider the area
     *
     * @return true if success, false otherwise
     */
    bool isTopLeftInside(const BoundingBox& bbox1, const BoundingBox& bbox2) const;

    /**
     * @return the bottom right corner of a given cv::RotatedRect
     */
    cv::Point2f getBottomRight(const cv::RotatedRect& rect) const;

    /**
     * @brief Make the start and end coordinates of a line uniform to our convention.
     *        The start point always has lower x value (opencv image convention); in case the x coordinates of
     *        start and end point match, make as start the point with higher y axis.
     *
     * @return a new line with updated coordinates if necessary, otherwise the given one
     */
    cv::Vec4i standardizeLine(const cv::Vec4i& line) const;

    /**
     * @brief Filters a set of lines based on specified criteria to remove redundant or irrelevant lines.
     *        The filters applied include:
     *        1. Eliminating short lines
     *        2. Removing lines with undesired angles
     *        3. Discarding lines in areas lacking sufficient white components.
     *        4. Excluding lines within the optional parking area.
     *        5. Eliminate lines that start or end near each other with similar angles, retaining the longest/strongest one.
     *        6. Removing perpendicular lines (only retaining lines with positive angles according to OpenCV convention).
     *
     * @param lines                       vector of lines to be filtered
     * @param referenceImage              image from which the lines were computed
     * @param parkingSpaceLinesAngles     vector with two pairs of allowable line angles
     * @param proximityThresholds         vector with two threshold values used for proximity checks; used for filters 5 and 6
     * @param minLength                   minimum allowable length of the lines; used in filter 1
     * @param angleThreshold              value to consider two angles similar
     * @param whitenessThreshold          minimum percentage of white pixels for a line; used in filter 3
     *
     * @return a vector containing only the lines that pass all the filters
     */
    std::vector<cv::Vec4i> filterLines(std::vector<cv::Vec4i>& lines, const cv::Mat& referenceImage, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                       const std::vector<double>& proximityThresholds, const double& minLength,
                                       const double& angleThreshold, const double& whitenessThreshold) const;

    /**
     *  @brief Matches the lines that form a parking space based on their geometric properties, such as angle similarity and proximity.
     *         The best candidate for each line is the one with the shortest valid distance.
     *
     *  @param finalLines                vector of lines to be matched
     *  @param parkingSpaceLinesAngles     vector with two pairs of allowable line angles
     *  @param startEndDistanceThreshold   maximum distance allowed between the start of one line and the end of another to be considered as a match
     *  @param endStartDistanceThreshold   maximum distance allowed between the end of one line and the start of another to be considered as a match
     *  @param angleThreshold              value to consider two angles similar
     *  @param deltaXThreshold             minimum horizontal distance required between two matched lines in the negative angle scenario
     *  @param deltaYThreshold             minimum vertical distance required between two matched lines in the positive angle scenario
     *
     *  @return a vector of paired lines, where each pair represents two lines that form a parking space
     */
    std::vector<std::pair<cv::Vec4i, cv::Vec4i>> matchLines(const std::vector<cv::Vec4i>& finalLines, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                                            const double& startEndDistanceThreshold, const double& endStartDistanceThreshold, const double& angleThreshold,
                                                            const double& deltaXThreshold, const double& deltaYThreshold) const;

    /**
     * @brief Converts matched line pairs into candidate rotated rectangles
     *
     * @param matchedLines vector of paired lines
     *
     * @returns a vector of cv::RotatedRect objects representing the candidate parking spaces generated from the matched lines.
     *
     * @details Given two lines, the center of the rotated rectangle is the mid point of the longest diagonal. The angle
     *          of the rotated rect is then the average of the line's angles.
     */
    std::vector<cv::RotatedRect> linesToRotatedRect(const std::vector<std::pair<cv::Vec4i, cv::Vec4i>>& matchedLines) const;

    /**
     * @brief Infers and generates new rotated rectangles based on filtering criteria and adjustments to existing rotated rectangles.
     *
     * @param rotatedRects           reference to a vector of cv::RotatedRect objects that will be filtered and updated with new candidate rectangles
     * @param parkingSpaceAngles     angle range of the rectangles
     */
    void InferRotatedRects(std::vector<cv::RotatedRect>& rotatedRects, std::pair<double, double> parkingSpaceAngles) const;

    /**
     * @brief Checks whether the left or right sides of a given rotated rectangle touch white areas on a binary mask.
     *
     * @param rotatedRect    rotated rectangle whose sides are to be checked
     * @param mask           binary mask
     * @param margin         margin distance to extend the lines beyond the rectangle's sides for collision detection
     * @param imgSize        size of the image
     *
     * @return a pair of booleans where:
     *         - The first boolean indicates whether the left side touches the white area on the mask.
     *         - The second boolean indicates whether the right side touches the white area on the mask.
     */
    std::pair<bool, bool> checkSides(const cv::RotatedRect& rotatedRect, const cv::Mat& mask, const int& margin, const cv::Size& imgSize) const;

    /**
     * @brief Removes rotated rectangles outliers from a list of detected parking spaces.
     *        The outliers are :
     *        1. rectangles between other rectangles ----> a parking space has always one side free for the car to enter
     *        2. rectangles with different aspect ratios (each type of rectangle angle has different base aspect ratios)
     *        3. overlapping rectangles
     *
     * @param rotatedRects                reference to a vector of cv::RotatedRect objects representing detected parking spaces that will be modified
     * @param parkingSpaceLinesAngles     vector of angle ranges representing the allowable angles for the parking space rectangles
     * @param imgSize                     size of the image
     * @param margin                      margin used when extending lines for collision detection with the mask
     * @param aspectRatioThresholds       vector containing thresholds for acceptable aspect ratios depending on the rectangle's angle
     */
    void removeOutliers(std::vector<cv::RotatedRect>& rotatedRects, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                        const cv::Size& imgSize, const int& margin, const std::vector<double>& aspectRatioThresholds) const;

    /**
     * @brief Groups rotated rects that represent the same parking space based on their center distance.
     *
     * @param boundingBoxesCandidates    candidate bounding boxes for parking spaces.
     * @param radius                     radius threshold (centers within this radius are considered to represent the same parking space)
     *
     * @return a vector of groups of parking space bounding boxes, where each inner vector contains rotated rects that represent the same parking space.
     */
    std::vector<std::vector<cv::RotatedRect>> matchParkingSpaces(std::vector<cv::RotatedRect>& boundingBoxesCandidates, const double& radius) const;

    /**
     * @brief Computes the average rotated rectangle for each group of rotated rectangles representing the same parking space.
     *
     * @param boundingBoxesParkingSpaces   vector of groups of cv::RotatedRect objects, where each inner vector
     *                                     contains bounding boxes that represent the same parking space
     *
     * @return a vector where each element is the average rectangle computed from a group of parking space bounding boxes
     */
    std::vector<cv::RotatedRect> computeAverageRect(const std::vector<std::vector<cv::RotatedRect>>& boundingBoxesParkingSpaces);

    /**
     * @brief Adjusts the perspective of cv::RotatedRect based on their position within the image (works only for a specific parking lot).
     *
     * @param rects                 vector of cv::RotatedRect objects to be modified
     * @param imgSize               size of the image
     * @param parkingSpaceAngles    vector of angle ranges representing the allowable angles for the parking space rectangles
     * @param margin                margin used when extending lines for collision detection with the mask
     * @param minIncrement          minimum adjustment value for the width or height of the rectangle based on its vertical position in the image
     * @param maxIncrement          maximum adjustment value for the width or height of the rectangle based on its vertical position in the image
     */
    void adjustPerspective(std::vector<cv::RotatedRect>& rects, const cv::Size& imgSize, const std::vector<std::pair<double, double>>& parkingSpaceAngles,
                           const int& margin, const unsigned short& minIncrement, const unsigned short& maxIncrement) const;
};

#endif