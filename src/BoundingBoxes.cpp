#include "../include/BoundingBoxes.hpp"

using namespace cv;
using namespace std;

const unsigned int CLOSE_KERNEL_SIZE = 9;
const unsigned int DILATE_KERNEL_SIZE = 5;
const unsigned int MIN_AREA_THRESHOLD = 6000;
const unsigned int MAX_AREA_THRESHOLD = 60000;
const unsigned int CIRCLE_NEIGHBORHOOD = 5;

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




cv::Mat BoundingBoxes::createROI(const cv::Mat& input) { // We focus the analysis of the image on the parking lots
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    // Define ROIs
    std::vector<cv::RotatedRect> rois;
    rois.push_back(cv::RotatedRect(cv::Point2f(572, 317), cv::Size2f(771, 282), 58));
    rois.push_back(cv::RotatedRect(cv::Point2f(950, 200), cv::Size2f(165, 710), -54));
    rois.push_back(cv::RotatedRect(cv::Point2f(1136, 105), cv::Size2f(73, 467), 118));

    std::vector<cv::RotatedRect> black_rois;        // More ad hoc ROI in order to refine the ROI selected
    black_rois.push_back(cv::RotatedRect(cv::Point2f(799, 343), cv::Size2f(1227, 125), 46));
    black_rois.push_back(cv::RotatedRect(cv::Point2f(326, 3), cv::Size2f(62, 113), 50));
    black_rois.push_back(cv::RotatedRect(cv::Point2f(861, 25), cv::Size2f(552, 64), 33));

    for (const auto& roiRect : rois) {
        cv::Point2f vertices[4];
        std::vector<cv::Point> contour;

        roiRect.points(vertices);    // Store the vertices of the ROI
        for (auto vertex : vertices) { contour.push_back(vertex); }
        cv::fillConvexPoly(mask, contour, cv::Scalar(255));
    }

    for (const auto& blackRoiRect : black_rois) {
        cv::Point2f vertices[4];
        std::vector<cv::Point> contour;

        blackRoiRect.points(vertices);
        for (auto vertex : vertices) { contour.push_back(vertex); }
        cv::fillConvexPoly(mask, contour, cv::Scalar(0));
    }

    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++)
            if (mask.at<uchar>(y, x) == 255)
                result.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(y, x);
    return result;
}



BoundingBoxes::BoundingBoxes(const cv::Mat &input) {
    int kernelSize = 5;
    int lowThreshold = 100;
    int ratio = 22;
    cv::Mat roiGray, roiCanny;


    // Image Preprocessing
    cv::Mat roiInput = createROI(input);        // Focus on the parking lots, my ROI


    // TODO might consider something with HSV, dont know yet
    //cv::Mat hsvRoiInput;
    //cv::cvtColor(roiInput, hsvRoiInput, cv::COLOR_BGR2HSV);

    auto gamma_correction = [](const cv::Mat& input) -> cv::Mat
    {
        // Gamma correction
        cv::Mat gc_image;
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i)
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 0.5) * 255.0);
        cv::LUT(input, lookUpTable, gc_image);

        return gc_image;
    };
    auto saturation_thresholding = [](const cv::Mat& input) -> cv::Mat
    {
        // Variables
        const unsigned int SATURATION_THRESHOLD = 30;

        // Convert to HSV
        cv::Mat hsv_image;
        cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
        cv::Mat saturation;
        cv::extractChannel(hsv_image, saturation, 1);

        // Thresholding
        cv::threshold(saturation, saturation, SATURATION_THRESHOLD, 255, cv::THRESH_BINARY);

        return saturation;
    };
    auto niBlack_thresholding = [](const cv::Mat& input) -> cv::Mat
    {
        // Variables
        const unsigned int NIBLACK_BLOCK_SIZE = 19;
        const double NIBLACK_K = 0.7;

        // Convert to grayscale
        cv::Mat gray_image;
        cv::cvtColor(input, gray_image, cv::COLOR_BGR2GRAY);

        // Thresholding
        cv::Mat niblack;
        cv::ximgproc::niBlackThreshold(gray_image, niblack, 255, cv::THRESH_BINARY, NIBLACK_BLOCK_SIZE, NIBLACK_K, cv::ximgproc::BINARIZATION_NIBLACK);

        return niblack;
    };
    auto fill_holes = [](cv::Mat& input) -> void
    {
        cv::Mat result = input.clone();

        // Fill holes
        cv::floodFill(result, cv::Point(0, 0), 255);
        cv::Mat inversed;
        cv::bitwise_not(result, inversed);
        input = (input | inversed);
    };
    cv::Mat image = roiInput.clone();
    cv::Mat gc_image = gamma_correction(image);
    cv::imshow("Gamma", gc_image);
    cv::waitKey(0);
    cv::Mat saturation = saturation_thresholding(gc_image);
    //cv::imshow("mongus", saturation);
    //cv::waitKey(0);
    cv::Mat niblack = niBlack_thresholding(image);
    cv::imshow("NI", niblack);
    cv::waitKey(0);


    cv::Mat mask = saturation & niblack;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)));
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)));
    fill_holes(mask);
    //cv::imshow("mongus", mask);
    //cv::waitKey(0);


    cvtColor( roiInput, roiGray, COLOR_BGR2GRAY );











    GaussianBlur(roiGray, roiGray, Size(kernelSize,kernelSize), 0);
    Canny( roiGray, roiCanny, lowThreshold, lowThreshold*ratio, kernelSize );


    cv::imshow("mongus", roiCanny);
    cv::waitKey(0);

    // MORFOLOGIAL OPERATION ----> ELIMINATE ALL THE IMAGE
    /*
    int erosion_size = 3;
    Mat element = getStructuringElement(MORPH_RECT,
                                        Size(erosion_size, erosion_size));
    cv::morphologyEx(roiCanny, roiCanny, cv::MORPH_CLOSE, element);
    */















    // Hough Transform
    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(roiCanny, hough_lines, 1, CV_PI /180, 45, 25, 30);


    cv::Mat hough_lines_image = input.clone();
    for (auto l : hough_lines) {
        cv::line(hough_lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0, 255));
    }

    img = hough_lines_image;


    // Other hough
    /*
    // Draw lines from lines1
    std::vector<cv::Vec2f> lines1;
    std::vector<cv::Vec2f> lines2;
    cv::HoughLines(roiCanny, lines1, 1, CV_PI/180, 75, 0, 0, CV_PI / 3, CV_PI / 2);
    cv::HoughLines(roiCanny, lines2, 1, CV_PI/180, 100, 0, 0, -CV_PI / 9, - CV_PI / 18);
    cv::Mat clone = input.clone();
    for (size_t i = 0; i < lines1.size(); i++) {
        float rho = lines1[i][0], theta = lines1[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;


        double length = 200;
        cv::Point pt1(cvRound(x0 + length * (-b)), cvRound(y0 + length * (a)));
        cv::Point pt2(cvRound(x0 - length * (-b)), cvRound(y0 - length * (a)));
        cv::line(clone, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // Red color for lines1
    }

    // Draw lines from lines2
    for (size_t i = 0; i < lines2.size(); i++) {
        float rho = lines2[i][0], theta = lines2[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(clone, pt1, pt2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA); // Blue color for lines2
    }

    */




    /*
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
    */
}




