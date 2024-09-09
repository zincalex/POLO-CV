#include "../include/Graphics.hpp"

void Graphics::getParkingRow(std::vector<cv::RotatedRect> &parkingSlots, int numParking, float angle, int xOffset,
                             float horizontalOffsetAdjustment, int yOffset, float verticalOffsetAdjustment,
                             int parkingWidth, int parkingHeight, int spacing, int extraSpacing, bool isDoubleRow,
                             bool isLowerSet) {
    float radianAngle = angle * CV_PI / 180.0;
    float verticalOffset = parkingWidth * std::sin(radianAngle);

    //Work with distinct parking space sets to manage different position, angle and size
    for (int i = 0; i < numParking; ++i) {
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
                cv::RotatedRect rotatedRectRight(centerRight, size, -angle);  // Angolo opposto per la parte inferiore
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


std::vector<cv::RotatedRect> Graphics::getBoxes() {
    std::vector<cv::RotatedRect> rectangles;

    // define parking slots properties
    int parkingWidth = 60;
    int parkingHeight = 120;
    int spacing = 20;

    // Section 1: (upper set, single row of parking spaces)
    int yOffsetTop = 80;
    int xOffsetTop = 320;  // Moves the spaces from the extreme left to the desired positions
    int numParkingTop = 5;
    float horizontalOffsetAdjustment = 0;
    float verticalOffsetAdjustment = 0;
    float angleTop = -30.0f;

    getParkingRow(rectangles, numParkingTop, angleTop, xOffsetTop, horizontalOffsetAdjustment, yOffsetTop, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0);

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
    getParkingRow(rectangles, numParkingMiddleTop, angleMiddleTop, xOffsetMiddle, horizontalOffsetAdjustment,yOffsetMiddle, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing,20,true, false);

    // Draw lower row of center set
    getParkingRow(rectangles, numParkingMiddleBottom, -angleMiddleBottom, xOffsetMiddle, horizontalOffsetAdjustment,yOffsetMiddle, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 20,true, true);

    // Section 3: (lower set, double row of parking spaces)
    int yOffsetBottom = yOffsetMiddle + parkingHeight * 2 + 50;
    int xOffsetBottom = 100;  // Inizio normale per la riga inferiore
    int numParkingBottom = 10;
    float angleBottom = 30.0f;

    horizontalOffsetAdjustment = 15;
    verticalOffsetAdjustment = -35;

    getParkingRow(rectangles, numParkingBottom, angleBottom, xOffsetBottom, horizontalOffsetAdjustment, yOffsetBottom, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0,true);

    getParkingRow(rectangles, numParkingBottom, -angleBottom, xOffsetBottom, horizontalOffsetAdjustment,yOffsetBottom, verticalOffsetAdjustment,parkingWidth, parkingHeight, spacing, 0,true, true);

    //vector with rectangles is reversed to match with how the parking space detector works
    std::reverse(rectangles.begin(), rectangles.end());
    return rectangles;
}


cv::Mat Graphics::drawMap(const std::vector<cv::RotatedRect> &parkingSpaces) {
    //define dimensions of original map
    int width = 950;
    int height = 750;

    cv::Mat parkingMap(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const auto & parkingSpace : parkingSpaces) {
        cv::Point2f vertices[4];
        parkingSpace.points(vertices);

        for (int i = 0; i < 4; i++) {
            cv::line(parkingMap, vertices[i], vertices[(i + 1) % 4], cv::Scalar(51, 36, 4), 20);
        }
        //std::string number = std::to_string(i);
        //cv::putText(parkingMap, number, parkingSpaces[i].center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);
    }
    return parkingMap;
}


void Graphics::fillRotatedRectsWithCar(cv::Mat &image, const std::vector<cv::RotatedRect> &rectangles,
                                       const std::vector<unsigned short> &carIndices) {
    for (size_t i = 0; i < rectangles.size(); ++i) {
        std::set<int> greenIndexesSet(carIndices.begin(), carIndices.end());
        // If index is found (car parked in the space) rectangle is filled with red or with blue if empty
        cv::Scalar color = (greenIndexesSet.find(i+1) != greenIndexesSet.end()) ? cv::Scalar(0, 0, 255) : cv::Scalar(130, 96, 21);

        cv::Point2f vertices[4];
        rectangles[i].points(vertices);

        std::vector<cv::Point> points;
        for (auto vertex : vertices) {
            points.push_back(vertex);
        }

        std::vector<std::vector<cv::Point>> fillPoints = { points };
        cv::fillPoly(image, fillPoints, color);
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

void Graphics::applyMap(const std::string &imageName, const std::vector<unsigned short> &occupiedParkingSpaces) {
    cv::Mat src = cv::imread(imageName);
    std::vector<cv::RotatedRect> rectangles = getBoxes();
    cv::Mat mapImage = drawMap(rectangles);
    fillRotatedRectsWithCar(mapImage, rectangles, occupiedParkingSpaces);
    mapOverlay(src, mapImage);
    cv::imshow("2DMap", src);
    cv::waitKey(0);
}
