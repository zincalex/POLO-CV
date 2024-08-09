#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load the image
    cv::Mat img = cv::imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence1/frames/2013-02-22_07_05_01.png");
    if (img.empty()) {
        std::cerr << "Could not open the image!" << std::endl;
        return -1;
    }

    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Apply mean shift filtering
    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(hsv, shifted, 20, 40);

    // Convert back to BGR color space
    cv::Mat bgr;
    cv::cvtColor(shifted, bgr, cv::COLOR_HSV2BGR);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // Apply thresholding to obtain a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours based on area and aspect ratio
    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (area > 500 && aspectRatio > 0.5 && aspectRatio < 3.0) {
            filteredContours.push_back(contour);
        }
    }

    // Draw the precise contours on the original image
    cv::Mat result = img.clone();
    cv::drawContours(result, filteredContours, -1, cv::Scalar(0, 255, 0), 2);

    // Display the results
    cv::imshow("Original Image", img);
    cv::imshow("Mean Shift Filtering", bgr);
    cv::imshow("Binary Image", binary);
    cv::imshow("Segmented Cars with Contours", result);

    cv::waitKey(0);
    return 0;
}