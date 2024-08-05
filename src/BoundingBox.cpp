#include "../include/BoundingBox.hpp"

using namespace cv;
using namespace std;

double computeAverageGrayscale(const Mat& gray, const RotatedRect& rect) {
    Mat mask = Mat::zeros(gray.size(), CV_8UC1);
    Point2f vertices[4];
    rect.points(vertices);
    vector<Point> pts;
    for (int i = 0; i < 4; i++) {
        pts.push_back(vertices[i]);
    }
    fillConvexPoly(mask, pts, Scalar(255));
    Scalar meanValue = mean(gray, mask);
    return meanValue[0];
}

BoundingBox::BoundingBox(const cv::Mat &input) {
    int kernelSize = 3;
    int lowThreshold = 26;
    int ratio = 10;

    cv::Mat input_gray, canny_img;
    cvtColor( input, input_gray, cv::COLOR_BGR2GRAY );
    GaussianBlur(input_gray, input_gray, cv::Size(kernelSize,kernelSize), 0);
    Canny( input_gray, canny_img, lowThreshold, lowThreshold*ratio, kernelSize );

    img = input.clone();
    /*std::vector<cv::Vec2f> lines1;
    std::vector<cv::Vec2f> lines2;
    cv::HoughLines(canny_img, lines1, 1, CV_PI/180, 150, 0, 0, CV_PI / 3, 1.395416); // 128 to select stronger lines
    cv::HoughLines(canny_img, lines2, 1, CV_PI/180, 150, 0, 0, -CV_PI / 9, - CV_PI / 18);
    */

    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(canny_img, hough_lines, 1, CV_PI /180, 50, 50, 10);
    /*
    // Function to draw lines
    auto drawLines = [](const std::vector<cv::Vec2f>& lines, cv::Mat& img, const cv::Scalar& color) {
        for (size_t i = 0; i < hough_lines.size(); i++) {
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
*/


    cv::Mat hough_lines_image = cv::Mat::zeros(img.size(), CV_8UC1);
    for (auto l : hough_lines) {
        cv::line(hough_lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,255,255));
    }

    cv::namedWindow("lesgoski");
    cv::imshow("lesgoski", hough_lines_image);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(hough_lines_image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat output = img.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        cv::RotatedRect rect = minAreaRect(contours[i]);

        // Ensure width and height are valid
        float minDim = std::min(rect.size.width, rect.size.height);
        float maxDim = std::max(rect.size.width, rect.size.height);
        if (minDim > 20 || maxDim / minDim < 2) {
            continue;
        }

        // Check if the average grayscale value is greater than 115
        double avgGrayscale = computeAverageGrayscale(input_gray, rect);
        if (avgGrayscale <= 115) {
            continue;
        }

        // Draw the rectangle if all conditions are met
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Debug output to check vertex values and thickness
        cout << "Drawing rectangle with vertices: ";
        for (int j = 0; j < 4; j++) {
            cout << "(" << vertices[j].x << ", " << vertices[j].y << ") ";
        }
        cout << endl;

        // Draw lines between the vertices
        for (int j = 0; j < 4; j++) {
            line(output, vertices[j], vertices[(j+1)%4], Scalar(0, 255, 0), 2);
        }

        // Create rotated bounding boxes that touch two of these lines
        // Choose two adjacent vertices to form a rotated bounding box
        Point2f boxPoints[4];
        for (int j = 0; j < 4; j++) {
            boxPoints[j] = vertices[j];
        }

        // Draw the rotated bounding box
        for (int j = 0; j < 4; j++) {
            line(output, boxPoints[j], boxPoints[(j+1)%4], Scalar(255, 0, 0), 2);
        }
        cv::namedWindow("sugoma");
        cv::imshow("sugoma", output);
    }
}


