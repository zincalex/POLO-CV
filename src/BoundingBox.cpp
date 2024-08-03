#include "../include/BoundingBox.hpp"



BoundingBox::BoundingBox(const cv::Mat &input) {
    int kernelSize = 3;
    int lowThreshold = 8;
    int ratio = 20;

    cv::Mat input_gray, canny_img;
    cvtColor( input, input_gray, cv::COLOR_BGR2GRAY );
    GaussianBlur(input_gray, input_gray, cv::Size(kernelSize,kernelSize), 0);
    Canny( input_gray, canny_img, lowThreshold, lowThreshold*ratio, kernelSize );

    img = input.clone();
    std::vector<cv::Vec2f> lines1;
    std::vector<cv::Vec2f> lines2;
    cv::HoughLines(canny_img, lines1, 1, CV_PI/180, 150, 0, 0, CV_PI / 3, 1.395416); // 128 to select stronger lines
    cv::HoughLines(canny_img, lines2, 1, CV_PI/180, 150, 0, 0, -CV_PI / 9, - CV_PI / 18);


    // Function to draw lines
    auto drawLines = [](const std::vector<cv::Vec2f>& lines, cv::Mat& img, const cv::Scalar& color) {
        for (size_t i = 0; i < lines.size(); i++) {
            float rho = lines[i][0];
            float theta = lines[i][1];
            double a = cos(theta);
            double b = sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;
            cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
            cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
            cv::line(img, pt1, pt2, color, 2, cv::LINE_AA);
        }
    };

    // Draw the lines on the image
    drawLines(lines1, img, cv::Scalar(0, 0, 255)); // Red lines

    drawLines(lines2, img, cv::Scalar(0, 255, 0)); // Green lines
}


