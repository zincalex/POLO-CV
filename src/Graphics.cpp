/**
 * @author Francesco Colla 2122543
 */
#include <set>
#include <opencv2/highgui.hpp>

#include "../include/Graphics.hpp"

void Graphics::applyMap(cv::Mat& src, const std::vector<unsigned short> &occupiedParkingSpaces) {
    // Get the rectangles that represent the map configuration
    std::vector<cv::RotatedRect> rectangles = Graphics::getBoxesForMap();
    cv::Mat mapImage = drawMap(rectangles);

    // Color the boxes accordingly
    fillRotatedRectsWithCar(mapImage, rectangles, occupiedParkingSpaces);

    // Apply the map on the image
    Graphics::mapOverlay(src, mapImage);
}


void Graphics::drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects) {
    cv::Scalar redColor(0, 0, 255);

    for (const cv::RotatedRect& rect : rotatedRects) {
        // Get the 4 vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Convert the vertices to integer points
        std::vector<cv::Point> intVertices(4);
        for (int i = 0; i < 4; i++) {
            intVertices[i] = vertices[i];
        }

        cv::polylines(image, intVertices, true, redColor, 2);  // Thickness of 2
    }
}


cv::Mat Graphics::maskApplication(cv::Mat &target, const cv::Mat &mask) {
    cv::Mat masked_image = target.clone();
    addWeighted(mask, 1, masked_image, 0.5, 0, masked_image);
    return masked_image;
}


void Graphics::createParkingRow(std::vector<cv::RotatedRect> &parkingSlots, int numParking, float angle, int xOffset,
                             float horizontalOffsetAdjustment, int yOffset, float verticalOffsetAdjustment,
                             int parkingWidth, int parkingHeight, int spacing, int extraSpacing, bool isDoubleRow,
                             bool isLowerSet) {

    float radianAngle = angle * CV_PI / 180.0;
    float verticalOffset = parkingWidth * std::sin(radianAngle);

    //Work with distinct parking space sets to manage different position, angle and size
    for (unsigned int i = 0; i < numParking; ++i) {
        if (isDoubleRow) {
            if (!isLowerSet) {
                // upper part of parking space set with two rows
                cv::Point2f centerLeft(xOffset + i * (parkingWidth + spacing + extraSpacing), yOffset);
                cv::Size2f size(parkingWidth, parkingHeight);
                cv::RotatedRect rotatedRectLeft(centerLeft, size, angle);
                parkingSlots.push_back(rotatedRectLeft);

            } else {
                // lower part of parking space set with two rows
                cv::Point2f centerRight(xOffset + i * (parkingWidth + spacing+extraSpacing) + horizontalOffsetAdjustment, yOffset + parkingHeight - verticalOffset + verticalOffsetAdjustment);
                cv::Size2f size(parkingWidth, parkingHeight);
                cv::RotatedRect rotatedRectRight(centerRight, size, -angle);  // Opposite angle for the inferior part
                parkingSlots.push_back(rotatedRectRight);
            }
        } else {
            // handles the single row of parking spaces on top
            cv::Point2f center(xOffset + i * (parkingWidth + spacing), yOffset);
            cv::Size2f size(parkingWidth, parkingHeight);
            cv::RotatedRect rotatedRect(center, size, angle);
            parkingSlots.push_back(rotatedRect);
        }
    }
}


std::vector<cv::RotatedRect> Graphics::getBoxesForMap() {
    std::vector<cv::RotatedRect> rectangles;

    // Define parking slots properties
    int parkingWidth = 60;
    int parkingHeight = 120;
    int spacing = 20;

    // Section 1: (upper set, single row of parking spaces)
    int yOffsetTop = 80;
    int xOffsetTop = 525;  // Moves the spaces from the extreme left to the desired positions
    int numParkingTop = 5;
    float horizontalOffsetAdjustment = 0;
    float verticalOffsetAdjustment = 0;
    float angleTop = -30.0f;

    createParkingRow(rectangles, numParkingTop, angleTop, xOffsetTop, horizontalOffsetAdjustment,
                     yOffsetTop, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0);

    // Section 2: (center set, double row of parking spaces with different angles, special parameters are used to handle the more extreme inclination)
    int yOffsetMiddle = yOffsetTop + parkingHeight + 60;
    int xOffsetMiddle = 120;
    int numParkingMiddleTop = 8;
    int numParkingMiddleBottom = 9;
    float angleMiddleTop = 45.0f;
    float angleMiddleBottom = -45.0f;
    horizontalOffsetAdjustment = -45;
    verticalOffsetAdjustment = 20;

    // Draw upper row of center set
    createParkingRow(rectangles, numParkingMiddleTop, angleMiddleTop, xOffsetMiddle, horizontalOffsetAdjustment,
                     yOffsetMiddle, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing,20,true, false);
    // Draw lower row of center set
    createParkingRow(rectangles, numParkingMiddleBottom, -angleMiddleBottom, xOffsetMiddle, horizontalOffsetAdjustment,
                     yOffsetMiddle, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 20,true, true);

    // Section 3: (lower set, double row of parking spaces)
    int yOffsetBottom = yOffsetMiddle + parkingHeight * 2 + 50;
    int xOffsetBottom = 100;  // Inizio normale per la riga inferiore
    int numParkingBottom = 10;
    float angleBottom = 30.0f;

    horizontalOffsetAdjustment = 15;
    verticalOffsetAdjustment = -35;

    createParkingRow(rectangles, numParkingBottom, angleBottom, xOffsetBottom, horizontalOffsetAdjustment, yOffsetBottom,
                     verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0,true);

    createParkingRow(rectangles, numParkingBottom, -angleBottom, xOffsetBottom, horizontalOffsetAdjustment,yOffsetBottom,
                     verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0,true, true);

    // Vector with rectangles is reversed to match with how the parking space detector works
    std::reverse(rectangles.begin(), rectangles.end());

    return rectangles;
}


cv::Mat Graphics::drawMap(const std::vector<cv::RotatedRect> &parkingSpaces) {
    // Define dimensions of original map
    int width = 950;
    int height = 750;

    cv::Mat parkingMap(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const cv::RotatedRect& parkingSpace : parkingSpaces) {
        cv::Point2f vertices[4];
        parkingSpace.points(vertices);

        for (unsigned int i = 0; i < 4; ++i)
            cv::line(parkingMap, vertices[i], vertices[(i + 1) % 4], cv::Scalar(51, 36, 4), 20);
    }

    return parkingMap;
}


void Graphics::fillRotatedRectsWithCar(cv::Mat &empty_map, const std::vector<cv::RotatedRect> &rectangles,
                                       const std::vector<unsigned short> &carIndices) {

    //define constants for better readability, colors read from project handout using gimp
    const cv::Scalar PARKING_OCCUPIED = cv::Scalar(0, 0, 255);
    const cv::Scalar PARKING_BUSY = cv::Scalar(130, 96, 21);
    const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
    const cv::Scalar WHITE = cv::Scalar(255, 255, 255);

    //since it was chosen to ignore the upper row of parking spaces, as allowed in the handout, mark the ignorable spaces in a unique way
    const cv::Scalar NON_CONSIDERED_COLOR = cv::Scalar(127,127,127);
    const std::string NON_CONSIDERED_TEXT = "N/A";
    const int PARKING_NUMBER = 37;

    //scanning through all the generated map rectangles
    for (unsigned int i = 0; i < rectangles.size(); ++i) {
        //transform in set for better searching of the index
        std::set<int> greenIndexesSet(carIndices.begin(), carIndices.end());

        //get the center position of the current rectangle to label it and create negative offset to improve style since angling the rect creates problems with perspective
        cv::Point label_position = rectangles[i].center;
        label_position.x -=20;
        // If index is found (car parked in the space) rectangle is filled with red or with blue if empty
        cv::Scalar color = (greenIndexesSet.find(i+1) != greenIndexesSet.end()) ? PARKING_OCCUPIED : PARKING_BUSY;
        color = (i > 36) ? NON_CONSIDERED_COLOR : color;

        //get the points to fill the rectangles with the correct color and fill them
        cv::Point2f vertices[4];
        rectangles[i].points(vertices);
        std::vector<cv::Point> points;
        for (cv::Point2f vertex : vertices)
            points.push_back(vertex);
        std::vector<std::vector<cv::Point>> fillPoints = { points };
        cv::fillPoly(empty_map, fillPoints, color);

        //apply labels to the rectangles, 1 is added to match the numeration of the parking spaces in detection
        std::string number = std::to_string(i+1);
        if (i < PARKING_NUMBER) {
            //if the number of the rectangle is within the range of the considered ones then the label with the number is applied
            cv::putText(empty_map, number, label_position, cv::FONT_HERSHEY_SIMPLEX, 1.2, BLACK, 4);
        } else {
            //if the rectangle is in the ignored ones the "N/A" label is applied, for better readability of the text a further offset is needed
            label_position.x -=15;
            cv::putText(empty_map, NON_CONSIDERED_TEXT, label_position, cv::FONT_HERSHEY_SIMPLEX, 1, BLACK, 4);
        }
    }

}


void Graphics::mapOverlay(cv::Mat &src, const cv::Mat& map) {
    cv::Mat mapLocal = map.clone();

    // Resize map for overlay
    cv::resize(mapLocal, mapLocal, cv::Size(270, 225));

    // Position of top left corner (read using image manipulation software and fixed)
    const int x_pos = 15;
    const int y_pos = 485;

    cv::Rect mapROI(x_pos, y_pos, mapLocal.cols, mapLocal.rows);
    mapLocal.copyTo(src(mapROI));
}








