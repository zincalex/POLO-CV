#include "../include/ParkingLotStatus.hpp"

cv::Mat createROI(const cv::Mat& image, const BoundingBox& bBox) {
    // Get the rotated rectangle from the bounding box
    cv::RotatedRect rotatedRect(bBox.getCenter(), bBox.getSize(), bBox.getAngle());

    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // Get the bounding rectangle of the rotated rectangle
    cv::Rect boundingRect = rotatedRect.boundingRect();

    // Define the destination points for perspective transformation
    cv::Point2f dstPoints[4] = {
            cv::Point2f(0, 0),                               // Top-left corner
            cv::Point2f(boundingRect.width - 1, 0),          // Top-right corner
            cv::Point2f(boundingRect.width - 1, boundingRect.height - 1), // Bottom-right corner
            cv::Point2f(0, boundingRect.height - 1)          // Bottom-left corner
    };

    // Get the perspective transformation matrix
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(vertices, dstPoints);

    // Apply the perspective transformation to get the warped image
    cv::Mat warpedImage;
    cv::warpPerspective(image, warpedImage, perspectiveMatrix, boundingRect.size());

    cv::Mat flippedImage;
    cv::flip(warpedImage, flippedImage, 0);  // 0 means flipping around the x-axis

    return flippedImage; // Return the warped (straightened) image
}


cv::Mat ParkingLotStatus::createMaskBlackishColors(const cv::Mat& image) const {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b bgrPixel = image.at<cv::Vec3b>(y, x);

            // Check if all BGR components are less than or equal to 30
            if (bgrPixel[0] <= 30 && bgrPixel[1] <= 30 && bgrPixel[2] <= 30) {
                mask.at<uchar>(y, x) = 255;
            } else {
                mask.at<uchar>(y, x) = 0;
            }
        }
    }
    return mask;
}


bool ParkingLotStatus::isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const {
    int whitePixels = cv::countNonZero(mask);
    double whitePixelPercentage = (double)whitePixels / totalPixels * 100.0;
    std::cout << "Percentage: " << whitePixelPercentage << std::endl;
    return whitePixelPercentage >= percentage;
}


void ParkingLotStatus::drawParkingLotStatus() {
    for (BoundingBox& bBox : bBoxes) {
        cv::Point2f vertices[4];
        bBox.getRotatedRect().points(vertices);

        cv::Scalar color;
        if (bBox.isOccupied())
            color = cv::Scalar(0, 0, 255);
        else
            color = cv::Scalar(255, 0, 0);

        for (int j = 0; j < 4; j++)
            cv::line(parkingImage, vertices[j], vertices[(j + 1) % 4], color, 2);

        cv::Point center = bBox.getCenter();
        cv::putText(parkingImage, std::to_string(bBox.getNumber()), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);

    }
}



ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes) {

    cv::Mat clone = parkingImage.clone();

    const double DEBUG = true;

    this->parkingImage = parkingImage;
    for (BoundingBox& bBox : bBoxes) {
        this->bBoxes.push_back(bBox);
        const unsigned int KERNEL_SIZE_CANNY = 5;
        const unsigned int LOW_THRESHOLD = 100;
        const unsigned int RATIO = 22;
        const double GAMMA = 1.25;
        const unsigned int SATURATION_THRESHOLD = 150;
        const unsigned int WHITENESS_THRESHOLD = 150;

        cv::Mat boxedInputImg = createROI(parkingImage, bBox);

        if (DEBUG) {
            cv::imshow("OG", boxedInputImg);
            std::cout << "PArking : " << bBox.getNumber() << std::endl;
        }
        // REAL METHOD
        // WHITE CHECK
        int totalPixels = boxedInputImg.rows * boxedInputImg.cols;
        cv::Mat whiteness;
        cv::extractChannel(boxedInputImg, whiteness, 1);
        cv::threshold(whiteness, whiteness, WHITENESS_THRESHOLD, 255, cv::THRESH_BINARY);
        if (DEBUG) cv::imshow("white", whiteness);

        if(isCar(whiteness, totalPixels, 22.0)) {
            if (DEBUG) std::cout << "is a car" << std::endl;
            bBox.updateState();
            this->bBoxes.push_back(bBox);
            if (DEBUG) cv::waitKey(0);
            continue;
        }
        else { // COLOR CHECK
            cv::Mat gc_image = ImageProcessing::gamma_correction(boxedInputImg, GAMMA);
            cv::Mat saturation = ImageProcessing::saturation_thresholding(gc_image, SATURATION_THRESHOLD);
            if (DEBUG) cv::imshow("sat", saturation);


            if(isCar(saturation, totalPixels, 15.0)) {
                if (DEBUG) std::cout << "is a car" << std::endl;
                bBox.updateState();
                this->bBoxes.push_back(bBox);
                if (DEBUG) cv::waitKey(0);
                continue;
            }
            else { // BLACK CHECK
                cv::Mat black = createMaskBlackishColors(boxedInputImg);
                if (DEBUG) cv::imshow("black", black);

                if(isCar(black, totalPixels, 10.0)) {
                    if (DEBUG) std::cout << "is a car" << std::endl;
                    bBox.updateState();
                    this->bBoxes.push_back(bBox);
                    if (DEBUG) cv::waitKey(0);
                    continue;
                }
                else { // FEATURE BASED
                    cv::Mat gray;
                    cv::cvtColor(boxedInputImg, gray, cv::COLOR_BGR2GRAY);

                    // Apply Gaussian blur
                    cv::Mat blurred;
                    cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0);

                    // Apply adaptive thresholding
                    cv::Mat adaptiveThresholdOutput;
                    cv::adaptiveThreshold(blurred, adaptiveThresholdOutput, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 9, 3);

                    // Find contours in the adaptive thresholded image
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(adaptiveThresholdOutput, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    // Create an empty mask to draw the filled contours
                    cv::Mat filledContours = cv::Mat::zeros(gray.size(), CV_8UC1);

                    // Fill the contours found
                    for (size_t i = 0; i < contours.size(); i++) {
                        cv::drawContours(filledContours, contours, (int)i, cv::Scalar(255), cv::FILLED);
                    }

                    cv::Mat img1_gray;
                    cv::Mat des1; // Descriptors
                    std::vector<cv::KeyPoint> kp1; // Keypoints

                    cv::cvtColor(boxedInputImg, img1_gray, cv::COLOR_BGR2GRAY);
                    cv::GaussianBlur(img1_gray, img1_gray, cv::Size(3,3), 0);

                    // Feature Detection and Description
                    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
                    sift->detectAndCompute(img1_gray, cv::noArray(), kp1, des1);

                    int numFeatures = kp1.size();
                    if (DEBUG) std::cout << "Number of features detected: " << numFeatures << std::endl;

                    // Draw the keypoints on the image
                    cv::Mat img_with_keypoints;
                    cv::drawKeypoints(boxedInputImg, kp1, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
                    if (DEBUG) cv::imshow("features", img_with_keypoints);


                    if(numFeatures >= 25) {
                        if (DEBUG) std::cout << "is a car" << std::endl;
                        bBox.updateState();
                        this->bBoxes.push_back(bBox);
                        if (DEBUG) cv::waitKey(0);
                        continue;
                    }
                    else {
                        this->bBoxes.push_back(bBox);
                        if (DEBUG) std::cout << "not a car" << std::endl;
                    }
                }

            }
        }
        if (DEBUG) std::cout << "-----------" << std::endl;
        if (DEBUG) cv::waitKey(0);
    }

}
