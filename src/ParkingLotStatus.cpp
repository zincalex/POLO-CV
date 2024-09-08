#include "../include/ParkingLotStatus.hpp"

bool ParkingLotStatus::isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const {
    int whitePixels = cv::countNonZero(mask);
    double whitePixelPercentage = (double)whitePixels / totalPixels * 100.0;
    return whitePixelPercentage >= percentage;
}


cv::Mat ParkingLotStatus::seeParkingLotStatus() {
    for (BoundingBox& bBox : bBoxes) {
        cv::Point2f vertices[4];
        bBox.getRotatedRect().points(vertices);

        cv::Scalar color;
        if (bBox.isOccupied())
            color = cv::Scalar(0, 0, 255);
        else
            color = cv::Scalar(255, 0, 0);

        for (unsigned int j = 0; j < 4; ++j)
            cv::line(parkingImage, vertices[j], vertices[(j + 1) % 4], color, 2);

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


ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes) {

    cv::Mat clone = parkingImage.clone();

    const double DEBUG = false;

    this->parkingImage = parkingImage;
    for (BoundingBox& bBox : bBoxes) {
        this->bBoxes.push_back(bBox);

        const double GAMMA = 1.25;
        const unsigned int SATURATION_THRESHOLD = 150;
        const unsigned int WHITENESS_THRESHOLD = 150;

        cv::Mat boxedInputImg = ImageProcessing::createROI(parkingImage, bBox);
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
                cv::Mat black = ImageProcessing::createMaskDarkColors(boxedInputImg);
                if (DEBUG) cv::imshow("black", black);

                if(isCar(black, totalPixels, 10.0)) {
                    if (DEBUG) std::cout << "is a car" << std::endl;
                    bBox.updateState();
                    this->bBoxes.push_back(bBox);
                    if (DEBUG) cv::waitKey(0);
                    continue;
                }
                else { // FEATURE BASED
                    cv::Mat img1_gray;
                    cv::Mat des1;
                    std::vector<cv::KeyPoint> kp1;

                    cv::Mat meanShiftImg;
                    cv::pyrMeanShiftFiltering(boxedInputImg, meanShiftImg, 20, 45, 3);

                    // Convert the result of mean shift to grayscale (SIFT needs grayscale input)
                    cv::Mat meanShiftGray;
                    cv::cvtColor(meanShiftImg, meanShiftGray, cv::COLOR_BGR2GRAY);

                    // Feature Detection and Description
                    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
                    sift->detectAndCompute(meanShiftGray, cv::noArray(), kp1, des1);

                    int numFeatures = kp1.size();
                    if (DEBUG) std::cout << "Number of features detected: " << numFeatures << std::endl;

                    // Draw the keypoints on the image
                    cv::Mat img_with_keypoints;
                    cv::drawKeypoints(boxedInputImg, kp1, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
                    if (DEBUG) cv::imshow("features", img_with_keypoints);
                    if (DEBUG) cv::imshow("mean shift", meanShiftImg);


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
