#include "../include/ParkingSpaceDetector.hpp"

double ParkingSpaceDetector::calculateLineLength(const cv::Vec4i& line) const {
    return std::sqrt(std::pow(line[2] - line[0], 2) + std::pow(line[3] - line[1], 2));
}


double ParkingSpaceDetector::calculatePointsDistance (const cv::Point& pt1, const cv::Point& pt2) const {
    return std::sqrt(std::pow(pt1.x - pt2.x, 2) + std::pow(pt1.y - pt2.y, 2));
}


double ParkingSpaceDetector::calculateLineAngle(const cv::Vec4i& line) const {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];

    return std::atan2(y2 - y1, x2 - x1) * 180 / CV_PI;
}


bool ParkingSpaceDetector::areAnglesSimilar(const double& angle1, const double& angle2, const double& angleThreshold) const {
    return std::abs(angle1 - angle2) < angleThreshold;
}


bool ParkingSpaceDetector::isInRange(const double& angle, const std::pair<double, double>& range) const {
    return angle >= range.first && angle <= range.second;
}


bool ParkingSpaceDetector::isWithinRadius(const cv::Point& center, const cv::Point& point, const double& radius) const {
    return std::sqrt(std::pow(center.x - point.x, 2) + std::pow(center.y - point.y, 2)) <= radius;
}


cv::Point2f ParkingSpaceDetector::getBottomRight(const cv::RotatedRect& rect) const {
    cv::Point2f vertices[4];
    rect.points(vertices);
    cv::Point2f bottomRight = vertices[0];
    double maxSum = bottomRight.x + bottomRight.y;

    // The bottom right corner has the highest sum of the x and y coordinate
    for (unsigned int i = 1; i < 4; ++i) {
        double sum = vertices[i].x + vertices[i].y;
        if (sum > maxSum) {
            maxSum = sum;
            bottomRight = vertices[i];
        }
    }

    return bottomRight;
}


cv::Vec4i ParkingSpaceDetector::standardizeLine(const cv::Vec4i& line) const {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

    if (x2 < x1){
        return cv::Vec4i(x2, y2, x1, y1);
    } else if (x2 == x1){
        if (y1 < y2) {
            return cv::Vec4i(x2, y2, x1, y1);
        }
    }

    return line;
}


std::vector<cv::Vec4i> ParkingSpaceDetector::filterLines(std::vector<cv::Vec4i>& lines, const cv::Mat& referenceImage, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                                         const std::vector<double>& proximityThresholds, const double& minLength,
                                                         const double& angleThreshold, const double& whitenessThreshold) const {

    const std::pair<double, double> FIRST_ANGLE_RANGE = parkingSpaceLinesAngles[0];
    const std::pair<double, double> SECOND_ANGLE_RANGE = parkingSpaceLinesAngles[1];

    std::vector<cv::Vec4i> filteredLines;
    std::vector<bool> keepLine(lines.size(), true);

    // Sort the lines by increasing value of x coordinates
    std::sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return a[0] < b[0];
    });

    // For each line
    for (unsigned int i = 0; i < lines.size(); ++i) {
        if (!keepLine[i]) continue; // Skip the line if already rejected

        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);
        double length1 = calculateLineLength(lines[i]);
        double angle1 = calculateLineAngle(lines[i]);

        // 1 CONTROL : eliminate short lines
        if (length1 < minLength) {
            keepLine[i] = false;
            continue;
        }


        // 2 CONTROL : mean average color around my line
        cv::Point2f center = (start1 + end1) * 0.5;
        cv::Size2f rectSize(length1, 3);    // Small width for the rectangle (e.g., 3 pixels)
        cv::RotatedRect rotatedRect(center, rectSize, angle1);
        cv::Rect boundingRect = rotatedRect.boundingRect();

        // Ensure the bounding rect is within the image boundaries
        boundingRect &= cv::Rect(0, 0, referenceImage.cols, referenceImage.rows);
        // Extract the region of interest (ROI) and create a mask for the rotated rectangle
        cv::Mat roi = referenceImage(boundingRect);
        cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);

        std::vector<cv::Point> points;
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        for (unsigned int z = 0; z < 4; z++)
            points.push_back(cv::Point(rectPoints[z].x - boundingRect.x, rectPoints[z].y - boundingRect.y));

        cv::fillConvexPoly(mask, points, cv::Scalar(255));
        // Calculate the mean color within the masked area
        cv::Scalar meanColor = cv::mean(roi, mask);
        // Calculate the percentage of whiteness
        double whiteness = (meanColor[0] + meanColor[1] + meanColor[2]) / (3.0 * 255.0);
        if (whiteness < 0.4) {
            keepLine[i] = false;
            continue;
        }


        // 3 CONTROL : eliminate lines with bad angles
        if (!((isInRange(angle1, FIRST_ANGLE_RANGE)) || (isInRange(angle1, SECOND_ANGLE_RANGE)))) {
            keepLine[i] = false;
            continue;
        }


        // Comparison with other lines
        for (unsigned int j = i + 1; j < lines.size(); ++j) {
            if (!keepLine[j]) continue;

            cv::Point2f start2(lines[j][0], lines[j][1]);
            cv::Point2f end2(lines[j][2], lines[j][3]);
            double length2 = calculateLineLength(lines[j]);
            double angle2 = calculateLineAngle(lines[j]);
            double startDistance = calculatePointsDistance(start1, start2);
            double endDistance = calculatePointsDistance(end1, end2);


            // 4 CONTROL : lines that start very close and end very close, with the same angle
            if ((startDistance < proximityThresholds[0] || endDistance < proximityThresholds[0]) && areAnglesSimilar(angle1, angle2, angleThreshold)) {
                keepLine[length1 >= length2 ? j : i] = false; // Keep the longest line
                if (length1 < length2)
                    break; // If we reject line i, it does not make sense to keep comparing
            }
        }
    }

    // Similar for loop done here because we need to have a more refined version of the lines to keep, different result if done in the previous for loop
    for (unsigned int i = 0; i < lines.size(); ++i) {
        if (!keepLine[i]) continue;

        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);
        double angle1 = calculateLineAngle(lines[i]);

        for (unsigned int j = 0; j < lines.size(); ++j) {
            if (!keepLine[j] || i == j) continue;

            cv::Point2f start2(lines[j][0], lines[j][1]);
            cv::Point2f end2(lines[j][2], lines[j][3]);
            double startDistance = calculatePointsDistance(start1, start2);
            double endDistance = calculatePointsDistance(end1, end2);
            double angle2 = calculateLineAngle(lines[j]);


            // 5 CONTROL : if end and start are close and angle is not similar discard it
            if ((startDistance <= proximityThresholds[1] || endDistance <= proximityThresholds[1]) && !areAnglesSimilar(angle1, angle2, angleThreshold)) {
                isInRange(angle1, SECOND_ANGLE_RANGE) ? keepLine[j] = false : keepLine[i] = false;
                break;
            }
        }
    }

    for (unsigned int i = 0; i < lines.size(); ++i) {
        if (keepLine[i])
            filteredLines.push_back(lines[i]);
    }

    return filteredLines;
}


std::vector<std::pair<cv::Vec4i, cv::Vec4i>> ParkingSpaceDetector::matchLines(const std::vector<cv::Vec4i>& linesSupreme, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                                                              const double& startEndDistanceThreshold, const double& endStartDistanceThreshold, const double& angleThreshold,
                                                                              const double& deltaXThreshold, const double& deltaYThreshold) const {

    std::vector<cv::Vec4i> lines = linesSupreme;
    // Sort the lines with decreasing y values
    std::sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return a[1] > b[1];
    });

    const std::pair<double, double> FIRST_ANGLE_RANGE = parkingSpaceLinesAngles[0];
    const std::pair<double, double> SECOND_ANGLE_RANGE = parkingSpaceLinesAngles[1];
    std::vector<std::pair<cv::Vec4i, cv::Vec4i>> matchedLines;

    /*cv::Mat image = imageSi.clone();*/
    for (unsigned int i = 0; i < lines.size(); ++i) {
        double angle1 = calculateLineAngle(lines[i]);
        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);

        std::optional<cv::Vec4i> bestCandidate;
        double minDist = std::numeric_limits<double>::max();

        // In order to visualize the line matched, modify the method and pass an image
        /*cv::line(image, cv::Point2f(start1.x, start1.y), cv::Point2f(end1.x, end1.y), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);*/

        for (unsigned int j = i + 1; j < lines.size(); ++j) {
            double angle2 = calculateLineAngle(lines[j]);

            if (areAnglesSimilar(angle1, angle2, angleThreshold)) {
                cv::Point2f start2(lines[j][0], lines[j][1]);
                cv::Point2f end2(lines[j][2], lines[j][3]);
                double startEndDist = calculatePointsDistance(start1, end2);
                double endStartDist = calculatePointsDistance(end1, start2);
                double absDeltaY = std::abs(start1.y - end2.y);
                double deltaX = start1.x - start2.x;

                // Control for positive rotated rects
                if (startEndDist <= startEndDistanceThreshold && isInRange(angle1, FIRST_ANGLE_RANGE) && absDeltaY >= deltaYThreshold && startEndDist < minDist) {
                    bestCandidate = lines[j];
                    minDist = startEndDist;
                    /*cv::line(image, cv::Point2f(start2.x, start2.y), cv::Point2f(end2.x, end2.y), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);*/
                } // Control for negative rotated rects
                else if (endStartDist <= endStartDistanceThreshold && isInRange(angle1, SECOND_ANGLE_RANGE) && deltaX >= deltaXThreshold && startEndDist < minDist) {
                    bestCandidate = lines[j];
                    minDist = startEndDist;
                    /*cv::line(image, cv::Point2f(start2.x, start2.y), cv::Point2f(end2.x, end2.y), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);*/
                }
            }
        }
        /*cv::imshow("progress", image);
        cv::waitKey(0);
        image = imageSi.clone();*/
        if (bestCandidate.has_value())
            matchedLines.push_back(std::make_pair(lines[i], bestCandidate.value()));
    }

    return matchedLines;
}


std::vector<cv::RotatedRect> ParkingSpaceDetector::linesToRotatedRect(const std::vector<std::pair<cv::Vec4i, cv::Vec4i>>& matchedLines) const {
    std::vector<cv::RotatedRect> rotatedRectCandidates;
    for (const std::pair<cv::Vec4i, cv::Vec4i>& pair : matchedLines) {
        cv::Vec4i line1 = pair.first;
        cv::Vec4i line2 = pair.second;

        // Extract start and end points from both lines
        cv::Point start1(line1[0], line1[1]);
        cv::Point end1(line1[2], line1[3]);
        cv::Point start2(line2[0], line2[1]);
        cv::Point end2(line2[2], line2[3]);

        // Calculate the maximum diagonal
        double diagonal1 = calculatePointsDistance(start1, end2);
        double diagonal2 = calculatePointsDistance(end1, start2);

        cv::Point diagonalStart, diagonalEnd;
        double diagonalLength;
        if (diagonal1 > diagonal2) {
            diagonalStart = start1;
            diagonalEnd = end2;
            diagonalLength = diagonal1;
        } else {
            diagonalStart = end1;
            diagonalEnd = start2;
            diagonalLength = diagonal2;
        }

        // Calculate the parameters of the rotated rect
        cv::Point center = (diagonalStart + diagonalEnd) * 0.5;

        double angle1 = calculateLineAngle(line1);
        double angle2 = calculateLineAngle(line2);
        double averageRotatedRectAngle = (90 + (angle1 + angle2) / 2.0);    // +90 in order to set right our notation

        double diagonalAngle = calculateLineAngle(cv::Vec4i(diagonalStart.x, diagonalStart.y, diagonalEnd.x, diagonalEnd.y));
        double angleDiff1 = std::abs(angle1 - diagonalAngle);
        double angleDiff2 = std::abs(angle2 - diagonalAngle);
        double averageAngleDiff = (angleDiff1 + angleDiff2) / 2.0;
        double angleDiffRad = averageAngleDiff * CV_PI / 180.0; // radiant conversion
        double width = std::abs(diagonalLength * std::sin(angleDiffRad));
        double height = std::abs(diagonalLength * std::cos(angleDiffRad));

        // Average rotated rect
        cv::RotatedRect rotatedRect(center, cv::Size2f(width, height), averageRotatedRectAngle);
        rotatedRectCandidates.push_back(rotatedRect);
    }

    return rotatedRectCandidates;
}

// TODO finist
void ParkingSpaceDetector::GenerateRotatedRects(std::vector<cv::RotatedRect>& rotatedRects) const {
    std::vector<cv::RotatedRect> filteredY;
    std::vector<cv::RotatedRect> filteredDegrees;
    std::vector<cv::RotatedRect> generatingRects;

    int max_X_Index = -1;
    double maxY = -1;
    double maxX = -1;

    // Loop to filter rectangles based on certain conditions
    for (size_t i = 0; i < rotatedRects.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(rotatedRects[i]);

        // Find the rectangle with the maximum Y value
        if (bottomRight.y > maxY) {
            maxY = bottomRight.y;
        }

        // Find the rectangle with the maximum X value
        if (bottomRight.x > maxX) {
            maxX = bottomRight.x;
            max_X_Index = i;
        }

        // Filter rectangles with angles between 0 and 50 degrees
        double angle = rotatedRects[i].angle;
        if (angle >= 0 && angle <= 50) {
            filteredDegrees.push_back(rotatedRects[i]);
        }

        // Filter rectangles based on Y and X values
        if (bottomRight.y > 460 && bottomRight.y < 520 && bottomRight.x > 750 && bottomRight.x < 860) {
            filteredY.push_back(rotatedRects[i]);
        }
    }

    // Remove the rectangle with the maximum Y value from the filteredY list, useful for further criterias
    for (size_t i = 0; i < filteredY.size(); ++i) {
        if (getBottomRight(filteredY[i]).y == maxY) {
            filteredY.erase(filteredY.begin() + i);
            break;
        }
    }

    // Add the rectangle with the maximum X value to the generatingRects list if it meets the condition
    if (maxX < 1250) {
        generatingRects.push_back(rotatedRects[max_X_Index]);
    }

    max_X_Index = -1;
    maxX = -1;

    // Find the rectangle with the maximum X value in filteredY
    for (size_t i = 0; i < filteredY.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(rotatedRects[i]);
        if (bottomRight.x > maxX) {
            maxX = bottomRight.x;
            max_X_Index = i;
        }
    }

    // Rectangle with the maximum X value in filteredY
    if (max_X_Index != -1) {
        generatingRects.push_back(filteredY[max_X_Index]);
    }


    double maxYValue = -1;
    int maxYIndex = -1;
    int minY_X_Range_Index = -1;
    double minY_Range = std::numeric_limits<double>::max();

    for (size_t i = 0; i < filteredDegrees.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(filteredDegrees[i]);

        // Find the rectangle with the maximum Y value and X > 700 with angle between 0 and 50 degree
        if (bottomRight.x > 700 && bottomRight.y > maxYValue) {
            maxYValue = bottomRight.y;
            maxYIndex = i;
        }

        // Find the rectangle with the minimum Y value and X within a specific range with angle between 0 and 50 degree
        if (bottomRight.y < minY_Range && bottomRight.x >= 685 && bottomRight.x <= 750) {
            minY_Range = bottomRight.x;
            minY_X_Range_Index = i;
        }
    }

    // Adds two "vertical" rectangles
    if (maxYIndex != -1 && minY_Range > 705) {
        generatingRects.push_back(filteredDegrees[maxYIndex]);
    }
    if (minY_X_Range_Index != -1) {
        generatingRects.push_back(filteredDegrees[minY_X_Range_Index]);
    }
    // generate new rotated rectangles
    for (const auto &rect: generatingRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        // bottom-right Y value is less than 200
        cv::Point2f bottomRight = getBottomRight(rect);
        if (bottomRight.y < 200) {
            // Calculate the translation to align the opposite side
            cv::Point2f midPointLongSide = (vertices[0] + vertices[1]) * 0.5;
            cv::Point2f translation = bottomRight - midPointLongSide;
            cv::Point2f newCenter = rect.center - translation;
            translation.x -= cv::norm(vertices[0] - bottomRight) / 1.1;
            translation.y -= cv::norm(vertices[1] - vertices[2]) / 2;
            newCenter += translation;

            cv::RotatedRect newRect(newCenter, rect.size, rect.angle);

            rotatedRects.push_back(newRect);
        } else {
            // Y >= 200,
            cv::Point2f midPointLongSide = (vertices[0] + vertices[1]) * 0.5;
            cv::Point2f translation = bottomRight - midPointLongSide;
            cv::Point2f newCenter = rect.center + translation;

            // Create the new rotated rectangle with the same angle and size
            cv::RotatedRect newRect(newCenter, rect.size, rect.angle);

            // Check if any side is greater than 90 and resize if needed
            if (newRect.center.x > 1150) {
                if (newRect.size.width > 110) {
                    newRect.size.width *= 0.55;
                }
                if (newRect.size.height > 110) {
                    newRect.size.height *= 0.55;
                }
                // After resizing, align vertex 0 with the midpoint of the side it was projected onto
                newRect.points(vertices);
                cv::Point2f newMidPointLongSide = (vertices[0] + vertices[1]) * 0.5;
                translation = newMidPointLongSide - vertices[0];
                newCenter = newRect.center - translation;
                newRect = cv::RotatedRect(newCenter, newRect.size, newRect.angle);
            }
            rotatedRects.push_back(newRect);
        }
    }
}


void ParkingSpaceDetector::removeOutliers(std::vector<cv::RotatedRect>& rotatedRects, const std::vector<std::pair<double, double>>& parkingSpaceLinesAngles,
                                          const cv::Size& imgSize, const int& margin, const std::vector<double>& aspectRatioThresholds) const {

    const std::pair<double, double> FIRST_ANGLE_RANGE = parkingSpaceLinesAngles[0];
    std::vector<bool> outlier(rotatedRects.size(), false);

    // Create the mask to see where the rotated rectangle are
    cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);
    for (const cv::RotatedRect& rect : rotatedRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        std::vector<cv::Point> verticesVector(4);
        for (unsigned int j = 0; j < 4; j++)
            verticesVector[j] = vertices[j];
        cv::fillPoly(mask, verticesVector, cv::Scalar(255));
    }


    // OUTLIER ELIMINATION
    // First outliers : parking spaces detected between other parking spaces
    for (unsigned int i = 0; i < rotatedRects.size(); ++i) {
        cv::Point2f vertices[4];
        rotatedRects[i].points(vertices);

        // Calculate the midpoints of the left and right sides
        cv::Point2f midpointLeft = (vertices[0] + vertices[3]) * 0.5;
        cv::Point2f midpointRight = (vertices[1] + vertices[2]) * 0.5;

        // Calculate the direction vectors (perpendicular to the rectangle's orientation)
        cv::Point2f direction = midpointRight - midpointLeft;
        direction = direction / cv::norm(direction);  // Normalize the direction vector

        // Calculate the endpoints for the lines
        cv::Point2f pointLeft = midpointLeft - margin * direction;
        cv::Point2f pointRight = midpointRight + margin * direction;

        // Remove rectangles that are between 2 parking spaces
        bool touchedLeft = false;
        bool touchedRight = false;
        int numSteps = 50; // Number of points to check along the line
        for (unsigned int step = 1; step <= numSteps; ++step) {
            double alpha = static_cast<double>(step) / numSteps;
            cv::Point2f interpolatedPointLeft = midpointLeft * (1.0f - alpha) + pointLeft * alpha;
            cv::Point2f interpolatedPointRight = midpointRight * (1.0f - alpha) + pointRight * alpha;

            // Ensure that the points are within the image boundaries
            if (interpolatedPointLeft.x >= 0 && interpolatedPointLeft.x < imgSize.width &&
                interpolatedPointLeft.y >= 0 && interpolatedPointLeft.y < imgSize.height &&
                interpolatedPointRight.x >= 0 && interpolatedPointRight.x < imgSize.width &&
                interpolatedPointRight.y >= 0 && interpolatedPointRight.y < imgSize.height) {

                // Draw yellow dots at each interpolated point, IMPORT THE IMAGE FOR REPORT FOR PRINTING
                /*cv::circle(image, interpolatedPointLeft, 3, cv::Scalar(0, 255, 255), 1); // Yellow dot
                cv::circle(image, interpolatedPointRight, 3, cv::Scalar(0, 255, 255), 1); // Yellow dot*/

                // Check if the interpolated points are in the white area on the mask
                if (mask.at<uchar>(cv::Point(interpolatedPointLeft)) == 255)
                    touchedLeft = true;

                if (mask.at<uchar>(cv::Point(interpolatedPointRight)) == 255)
                    touchedRight = true;
            }

            if (touchedLeft && touchedRight)
                outlier[i] = true;  // Mark the rectangle as an outlier
            /*cv::line(image, pointLeft, pointRight, cv::Scalar(0, 255, 0), 2);  // Green line with thickness 2*/
        }


        // Second outliers : rotated rects with a specific angles that differ to much from the other types
        double angle = rotatedRects[i].angle - 90;
        double width = rotatedRects[i].size.width;
        double height = rotatedRects[i].size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // Depending on the angle different aspect ratio are checked
        if (isInRange(angle, FIRST_ANGLE_RANGE)) {
            if (aspectRatio < aspectRatioThresholds[0]) {
                outlier[i] = true;
            }
        }
        else {
            if (aspectRatio > aspectRatioThresholds[1])
                outlier[i] = true;
        }
    }


    // Third outliers: remove overlapping rotated rects
    for (unsigned int i = 0; i < rotatedRects.size(); ++i) {    // We need the rects to be filtered, if this operation is done in the previous for loop returns we achive different (worse) results
        if (outlier[i]) continue;

        for (unsigned int j = i + 1; j < rotatedRects.size(); ++j) {
            if (outlier[j]) continue;

            int smallRectIndex = i;
            int largeRectIndex = j;
            if (rotatedRects[i].size.area() > rotatedRects[j].size.area()) {
                smallRectIndex = j;
                largeRectIndex = i;
            }

            // Find intersection points between the two rectangles
            std::vector<cv::Point2f> intersectionPoints;
            int result = cv::rotatedRectangleIntersection(rotatedRects[smallRectIndex], rotatedRects[largeRectIndex], intersectionPoints);

            if (result == cv::INTERSECT_FULL || result == cv::INTERSECT_PARTIAL) {
                // Calculate the intersection area (Polygon area)
                double intersectionArea = cv::contourArea(intersectionPoints);

                // Compare with the smaller rectangle's area
                double smallRectArea = rotatedRects[smallRectIndex].size.area();

                // If intersection area is close to the smaller rectangle's area, keep the smaller rectangle
                if ((intersectionArea / smallRectArea) > 0.8)
                    outlier[largeRectIndex] = true;
            }
        }
    }

    auto it = std::remove_if(rotatedRects.begin(), rotatedRects.end(),[&](const cv::RotatedRect& rect) {
        unsigned int index = &rect - &rotatedRects[0];
        return outlier[index];
    });
    rotatedRects.erase(it, rotatedRects.end());
}


std::vector<cv::RotatedRect> ParkingSpaceDetector::computeAverageRect(const std::vector<std::vector<cv::RotatedRect>>& boundingBoxesParkingSpaces) {

    std::vector<cv::RotatedRect> averages;
    for (const std::vector<cv::RotatedRect>& parkingSpace : boundingBoxesParkingSpaces) {
        unsigned int sumXCenter = 0, sumYCenter = 0;
        unsigned int sumWidth = 0, sumHeight = 0;
        unsigned int sumAngles = 0;
        unsigned int count = parkingSpace.size();
        double avgAngleSin = 0.0;
        double avgAngleCos = 0.0;

        for (const cv::RotatedRect& box : parkingSpace) {
            sumXCenter += box.center.x;
            sumYCenter += box.center.y;
            sumWidth += box.size.width;
            sumHeight += box.size.height;

            double angleRad = box.angle * CV_PI / 180.0;
            avgAngleSin += std::sin(angleRad);
            avgAngleCos += std::cos(angleRad);
        }

        cv::Point avgCenter(static_cast<int>(sumXCenter / count), static_cast<int>(sumYCenter / count));
        cv::Size avgSize = cv::Size(static_cast<int>(sumWidth / count), static_cast<int>(sumHeight / count));

        double avgAngleRad = std::atan2(avgAngleSin / count, avgAngleCos / count);
        double avgAngle = avgAngleRad * 180.0f / CV_PI;                                     // back to degree

        cv::RotatedRect avgRotRect (avgCenter, avgSize, avgAngle);
        averages.push_back(avgRotRect);
    }

    return averages;
}



// TODO eliminate
void drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects) {
    // Define the color for the border (Red)
    cv::Scalar redColor(0, 0, 255);  // BGR format, so (0, 0, 255) is red

    for (const cv::RotatedRect& rect : rotatedRects) {
        // Get the 4 vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Convert the vertices to integer points (required by polylines)
        std::vector<cv::Point> intVertices(4);
        for (int i = 0; i < 4; i++) {
            intVertices[i] = vertices[i];
        }

        // Draw the rectangle with a red border
        cv::polylines(image, intVertices, true, redColor, 2);  // Thickness of 2
    }
}





ParkingSpaceDetector::ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir) {
    // PARAMETERS
    // Main angle ranges of the parking lines (can be adjusted accordingly)
    const std::vector<std::pair<double, double>> PARKING_SPACE_LINES_ANGLES = {std::make_pair(5.0, 20.0), std::make_pair(-87, -55.0)};

    // Line filter parameters
    const std::vector<double> PROXIMITY_THRESHOLDS = {25.0, 15.0};    // proximity distance to consider 2 lines close, (angle dependant)
    const double MIN_LINE_LENGTH = 20;
    const double ANGLE_THRESHOLD = 20.0;                              // difference to consider 2 angles similar
    const double WHITENESS_THRESHOLD = 0.4;                           // used for eliminate some lines

    // Matching lines parameters
    const double START_END_DISTANCE_THRESHOLD = 85.0;                 // max distance between the start of the first line and the end of the second line
    const double END_START_DISTANCE_THRESHOLD = 120.0;                // max distance between the end of the first line and the start of the second line
    const double DELTA_X_THRESHOLD = 20.0;
    const double DELTA_Y_THRESHOLD = 15.0;
    const double MAX_PARALLEL_ANGLE = 10.0;                           // max angle to consider two lines as parallel

    // Outliers elimination parameters
    const std::vector<double> ASPECT_RATIO_THRESHOLDS = {1.4, 1.9};   // minimum aspect ratio for the rotated rects based on the angle of the parking space
    const int MARGIN = 90;                                            // distance to consider from the edges of a rotated rect

    // Other parameters
    const double RADIUS = 35.0;
    const float IOU_THRESHOLD = 0.9;

    std::vector<cv::RotatedRect> boundingBoxesCandidates;
    for (const auto& iter : std::filesystem::directory_iterator(emptyFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat input = cv::imread(imgPath);
        if (input.empty()) {
            std::cerr << "Error opening the image" << std::endl;
        }
        cv::Size imgSize = input.size();
        cv::Mat clone = input.clone(); // TODO eliminate

        // LSH line detector
        cv::Mat gray;
        std::vector<cv::Vec4i> lines;
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        lsd->detect(gray, lines);

        // Make the start and end of the lines uniform
        for (size_t i = 0; i < lines.size(); ++i)
            lines[i] = standardizeLine(lines[i]);

        // Filter the lines
        std::vector<cv::Vec4i> filteredLines = filterLines(lines, input, PARKING_SPACE_LINES_ANGLES, PROXIMITY_THRESHOLDS, MIN_LINE_LENGTH, ANGLE_THRESHOLD, WHITENESS_THRESHOLD);
        std::vector<std::pair<cv::Vec4i, cv::Vec4i>> matchedLines = matchLines(filteredLines, PARKING_SPACE_LINES_ANGLES, START_END_DISTANCE_THRESHOLD, END_START_DISTANCE_THRESHOLD, MAX_PARALLEL_ANGLE, DELTA_X_THRESHOLD, DELTA_Y_THRESHOLD);

        // Build the rotated rects
        std::vector<cv::RotatedRect> rotatedRects = linesToRotatedRect(matchedLines);

        // Infer missing rects
        GenerateRotatedRects(rotatedRects);

        // Remove rotated rects outliers
        removeOutliers(rotatedRects, PARKING_SPACE_LINES_ANGLES, imgSize, MARGIN, ASPECT_RATIO_THRESHOLDS);


        /*// TODO Eliminate
        drawRotatedRects(clone, rotatedRects);
        cv::imshow("Rotated Rectangles", clone);
        cv::waitKey(0);*/


        boundingBoxesCandidates.insert(boundingBoxesCandidates.end(), rotatedRects.begin(), rotatedRects.end());
    }


    std::vector<std::vector<cv::RotatedRect>> boundingBoxesParkingSpaces;
    while (!boundingBoxesCandidates.empty()) {
        std::vector<cv::RotatedRect> parkingSpaceBoxes;

        // First populate the map with the first not analyzed parking space
        auto iterator = boundingBoxesCandidates.begin();
        cv::Point centerParkingSpace = iterator->center;
        parkingSpaceBoxes.push_back(*iterator);
        boundingBoxesCandidates.erase(iterator); // remove it in order to not insert it twice

        // Look for all the other candidates if there is one that represent the same parking lot
        auto iterator2 = boundingBoxesCandidates.begin();
        while (iterator2 != boundingBoxesCandidates.end()) {
            cv::Point anotherCenter = iterator2->center;

            if (isWithinRadius(centerParkingSpace, anotherCenter, RADIUS)) {
                parkingSpaceBoxes.push_back(*iterator2);
                iterator2 = boundingBoxesCandidates.erase(iterator2);  // Erase and get the next iterator
            } else {
                ++iterator2;  // Pre-increment for efficiency purpose
            }
        }

        boundingBoxesParkingSpaces.push_back(parkingSpaceBoxes);
    }

    // For all valid boxes, make the average (since we have no score for a bounding box, all have the same weight)
    std::vector<cv::RotatedRect> finalBoundingBoxes = computeAverageRect(boundingBoxesParkingSpaces);

    // Build the bounding boxes
    unsigned short parkNumber = 1;
    for (const cv::RotatedRect rotRect : finalBoundingBoxes) {
        BoundingBox bbox = BoundingBox(rotRect, parkNumber++);
        bBoxes.push_back(bbox);
    }
}