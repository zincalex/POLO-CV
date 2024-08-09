#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "filesystem"

//very good but not in scope

// Helper function to load image sequence
std::vector<cv::Mat> loadImages(const std::string& pathSequenceFramesDir) {
    std::vector<cv::Mat> images;
    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat inputImg = cv::imread(imgPath);
        if (inputImg.empty()) {
            std::cout << "Error opening the image" << std::endl;
            exit(-1);
        }

        cv::Mat ycrcb_image;
        cvtColor(inputImg, ycrcb_image, cv::COLOR_BGR2YCrCb);

        // Split the image into separate channels
        std::vector<cv::Mat> channels;
        split(ycrcb_image, channels);

        // Apply Histogram Equalization to the Y channel (intensity)
        equalizeHist(channels[0], channels[0]);

        // Merge the channels back
        cv::Mat equalized_image;
        merge(channels, equalized_image);

        // Convert the image back from YCrCb to BGR
        cvtColor(equalized_image, equalized_image, cv::COLOR_YCrCb2BGR);

        images.push_back(inputImg);
    }
    return images;
}

int main() {
    // Load the sequence of images (replace with your image path, extension, and range)
    std::string path = "../ParkingLot_dataset/sequence1/frames"; // Example: "/path/to/your/images/img"
    std::string extension = ".png"; // Example: ".png"
    int startFrame = 1;
    int endFrame = 10;
    std::vector<cv::Mat> images = loadImages(path);
    if (images.empty()) {
        std::cerr << "No images to process" << std::endl;
        return -1;
    }

    // Create Background Subtractor object
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();

    cv::Mat fgMask;

    for (const auto& frame : images) {
        // Apply background subtraction
        pBackSub->apply(frame, fgMask);

        // Morphological operations to clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
        //cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter contours based on area
        std::vector<std::vector<cv::Point>> filteredContours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 500) {  // Adjust this threshold as necessary
                filteredContours.push_back(contour);
            }
        }

        // Draw the contours on the original frame
        cv::Mat result = frame.clone();
        cv::drawContours(result, filteredContours, -1, cv::Scalar(0, 255, 0), 2);

        // Display the results
        cv::imshow("Original Frame", frame);
        cv::imshow("Foreground Mask", fgMask);
        cv::imshow("Segmented Cars", result);

        // Wait for a key press to proceed to the next frame
        cv::waitKey();
    }

    cv::destroyAllWindows();
    return 0;
}
