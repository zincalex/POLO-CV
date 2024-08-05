#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat img, result, mask, edges, output;
int threshold_value = 53;

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

int main() {
    // Load the image
    img = cv::imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence0/frames/2013-02-24_10_05_04.jpg");
    if (img.empty()) {
        std::cout << "Error opening the image" << std::endl;
        return -1;
    }

    // Define the ROI
    std::vector<cv::RotatedRect> rois;
    rois.push_back(cv::RotatedRect(cv::Point2f(572, 317), cv::Size2f(771, 282), 58));  // Adjust the position and angle as needed
    rois.push_back(cv::RotatedRect(cv::Point2f(950, 200), cv::Size2f(165, 710), -54));  // Adjust the position and angle as needed
    rois.push_back(cv::RotatedRect(cv::Point2f(1136, 105), cv::Size2f(73, 467), 118));  // Adjust the position and angle as needed

    std::vector<cv::RotatedRect> black_rois;
    black_rois.push_back(cv::RotatedRect(cv::Point2f(799, 343), cv::Size2f(1227, 125), 46));  // Adjust the position and angle as needed
    black_rois.push_back(cv::RotatedRect(cv::Point2f(326, 3), cv::Size2f(62, 113), 50));  // Adjust the position and angle as needed
    black_rois.push_back(cv::RotatedRect(cv::Point2f(861, 25), cv::Size2f(552, 64), 33));  // Adjust the position and angle as needed

    mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // Create mask for the ROIs
    for (const auto& roiRect : rois) {
        cv::Point2f vertices[4];
        roiRect.points(vertices);
        std::vector<cv::Point> contour;
        for (int j = 0; j < 4; j++) {
            contour.push_back(vertices[j]);
        }
        cv::fillConvexPoly(mask, contour, cv::Scalar(255));
    }

    // Affine the ROIs using black rotatedRects
    for (const auto& blackRoiRect : black_rois) {
        cv::Point2f vertices[4];
        blackRoiRect.points(vertices);
        std::vector<cv::Point> contour;
        for (int j = 0; j < 4; j++) {
            contour.push_back(vertices[j]);
        }
        cv::fillConvexPoly(mask, contour, cv::Scalar(0));
    }

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

    return 0;
}