#include <iostream>
#include <filesystem>

#include "../include/BoundingBoxes.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/highgui.hpp"


cv::Mat result, mask, edges, output;
int threshold_value = 53;

/*
void on_trackbar(int, void*) {
    // Apply the threshold to create a binary mask
    result = img.clone();
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (mask.at<uchar>(y, x) == 255 && cv::mean(result.at<cv::Vec3b>(y, x))[0] < threshold_value) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Set pixel to black
            }
        }
    }

    // Now apply Gaussian blur to the result image
    cv::Mat blurred;
    cv::GaussianBlur(result, blurred, cv::Size(5, 5), 0);

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
    //cv::cvtColor(blurred, gray, cv::COLOR_BGR2HSV);
    //cv::imshow("HSV", gray);

    // Detect edges using Canny
    cv::Canny(gray, edges, 50, 150);

    // Apply Hough Line Transform to detect lines
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);
    
    // Draw the lines on the result image
    output = result.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(output, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Display the images
    cv::imshow("Thresholded Image", result);
    cv::imshow("Edges", edges);
    cv::imshow("Detected Parking Lines", output);
}
*/
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <frames_directory>" << std::endl;
        return -1;
    }

    std::filesystem::path pathSequenceFramesDir = std::filesystem::absolute(argv[1]);
    if (!std::filesystem::exists(pathSequenceFramesDir) || !std::filesystem::is_directory(pathSequenceFramesDir)) {
        std::cerr << "Input directory does not exist or is not a directory." << std::endl;
        return -1;
    }

    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat inputImg = cv::imread(imgPath);
        if (inputImg.empty()) {
            std::cout << "Error opening the image" << std::endl;
            return -1;
        }
        BoundingBoxes BBoxes = BoundingBoxes(inputImg);

        cv::Mat test = BBoxes.getImg();
        cv::namedWindow("mongus", cv::WINDOW_AUTOSIZE);
        //cv::imshow("mongus", test);
        //cv::waitKey(0);
    }



    /*
    // Create windows
    cv::namedWindow("Thresholded Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Edges", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Detected Parking Lines", cv::WINDOW_AUTOSIZE);

    // Create trackbar
    cv::createTrackbar("Threshold", "Thresholded Image", &threshold_value, 255, on_trackbar);

    // Initial call to display the result
    on_trackbar(threshold_value, 0);

    // Wait until user exits the program
    cv::waitKey(0);
    */
    return 0;
}