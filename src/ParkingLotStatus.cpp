/**
 * @author Alessandro Viespoli 2120824
 */
#include "../include/ParkingLotStatus.hpp"
#include "../include/ImageProcessing.hpp"

#include <opencv2/features2d.hpp>

ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes, const cv::Mat& segmentationMaskHSV) {
    // Constant parameter only used during initialization
    const double GAMMA = 1.25;
    const unsigned int SATURATION_THRESHOLD = 150;
    const unsigned int WHITENESS_THRESHOLD = 150;

    this->parkingImage = parkingImage;

    for (BoundingBox& bBox : bBoxes) { // For each parking space
        // Crop the image around the bounding box and obtain the ROI of the parking space
        cv::Mat boxedInputImg = ImageProcessing::createROI(parkingImage, bBox);           // BGR
        cv::Mat hsvBoxedSegMask = ImageProcessing::createROI(segmentationMaskHSV, bBox);  // binary

        // SEGMENTATION MASK CHECK
        int totalPixels = boxedInputImg.rows * boxedInputImg.cols;
        if(isCar(hsvBoxedSegMask, totalPixels, 26)) {  // 26% of the pixels must be white in order to be a car
            bBox.updateState();
            continue;  // next bounding box
        }
        else { // COLORED CARS CHECK, sometimes the segmentation has some difficulty catching the colored car when their color is not enough bright
            cv::Mat gc_image = ImageProcessing::gamma_correction(boxedInputImg, GAMMA);
            cv::Mat saturation = ImageProcessing::saturation_thresholding(gc_image, SATURATION_THRESHOLD);

            if (isCar(saturation, totalPixels, 15.0)) {  // 15% of the pixels must be white in order to be a car
                bBox.updateState();
                continue;  // next bounding box
            }
            else { // BLACK CARS CHECK
                cv::Mat black = ImageProcessing::createMaskDarkColors(boxedInputImg);

                if (isCar(black, totalPixels, 10.0)) {  // 10% of the pixels must be white in order to be a car
                    bBox.updateState();
                    continue;  // next bounding box
                }
                else { // FEATURES CHECK
                    cv::Mat descriptor;
                    std::vector<cv::KeyPoint> keypoints;

                    cv::Mat meanShiftImg;
                    cv::pyrMeanShiftFiltering(boxedInputImg, meanShiftImg, 20, 45, 3);

                    // Convert to grayscale (SIFT needs grayscale input)
                    cv::Mat meanShiftGray;
                    cv::cvtColor(meanShiftImg, meanShiftGray, cv::COLOR_BGR2GRAY);

                    // Feature Detection and Description
                    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
                    sift->detectAndCompute(meanShiftGray, cv::noArray(), keypoints, descriptor);

                    int numFeatures = keypoints.size();
                    if (numFeatures >= 35) {  // at least 35 keypoints in order to be a car
                        bBox.updateState();
                        continue;
                    }
                }
            }
        }
    }

    // Insert the updated bounding boxes
    this->bBoxes = bBoxes;
}


cv::Mat ParkingLotStatus::seeParkingLotStatus() {
    for (BoundingBox& bBox : bBoxes) { // For each parking space
        cv::Point2f vertices[4];
        bBox.getRotatedRect().points(vertices);

        cv::Scalar color;
        // Based on the occupancy, change the color
        bBox.isOccupied() ? color = cv::Scalar(0, 0, 255) : color = cv::Scalar(255, 0, 0);


        // Draw the bounding boxe lines
        for (unsigned int j = 0; j < 4; ++j)
            cv::line(parkingImage, vertices[j], vertices[(j + 1) % 4], color, 2);

        // Put the parking space number at the center
        cv::Point center = bBox.getCenter();
        cv::putText(parkingImage, std::to_string(bBox.getNumber()), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }

    return parkingImage;
}


std::vector<unsigned short> ParkingLotStatus::getOccupiedParkingSpaces() const {
    std::vector<unsigned short> occupiedPS;
    for (const BoundingBox& bBox : bBoxes)
        if (bBox.isOccupied())
            occupiedPS.push_back(bBox.getNumber());

    return occupiedPS;
}


bool ParkingLotStatus::isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const {
    int whitePixels = cv::countNonZero(mask);
    double whitePixelPercentage = static_cast<double>(whitePixels) / totalPixels * 100.0;
    return whitePixelPercentage >= percentage;
}