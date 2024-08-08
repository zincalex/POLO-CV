#include "../include/BoundingBoxes.hpp"


const unsigned int CLOSE_KERNEL_SIZE = 9;
const unsigned int DILATE_KERNEL_SIZE = 5;
const unsigned int MIN_AREA_THRESHOLD = 6000;
const unsigned int MAX_AREA_THRESHOLD = 60000;
const unsigned int CIRCLE_NEIGHBORHOOD = 5;


cv::Mat BoundingBoxes::createROI(const cv::Mat& input) { // We focus the analysis of the image on the parking lots
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    // Define ROIs
    std::vector<cv::RotatedRect> rois;
    rois.push_back(cv::RotatedRect(cv::Point2f(580, 317), cv::Size2f(771, 282), 58));
    rois.push_back(cv::RotatedRect(cv::Point2f(950, 192), cv::Size2f(165, 710), 128));
    rois.push_back(cv::RotatedRect(cv::Point2f(1084, 83), cv::Size2f(452, 54), 28));
    //rois.push_back(cv::RotatedRect(cv::Point2f(950, 200), cv::Size2f(165, 710), -54));
    //rois.push_back(cv::RotatedRect(cv::Point2f(1136, 105), cv::Size2f(73, 467), 118));

    std::vector<cv::RotatedRect> black_rois;        // More ad hoc ROI in order to refine the ROI selected
    black_rois.push_back(cv::RotatedRect(cv::Point2f(777, 343), cv::Size2f(1227, 125), 47));
    //black_rois.push_back(cv::RotatedRect(cv::Point2f(326, 3), cv::Size2f(62, 113), 50));  Parking lot high left
    //black_rois.push_back(cv::RotatedRect(cv::Point2f(861, 25), cv::Size2f(552, 64), 33)); Old measures
    black_rois.push_back(cv::RotatedRect(cv::Point2f(861, 30), cv::Size2f(1042, 72), 32));

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

cv::Mat BoundingBoxes::gamma_correction(const cv::Mat& input, const double& gamma) {
    cv::Mat img_float, img_gamma;

    input.convertTo(img_float, CV_32F, 1.0 / 255.0);    // Convert to float and scale to [0, 1]
    cv::pow(img_float, gamma, img_gamma);               // Gamma correction
    img_gamma.convertTo(img_gamma, CV_8UC3, 255.0);       // Convert back to 8-bit type

    CV_Assert(img_gamma.type() == CV_8UC3);
    return img_gamma;
}

cv::Mat BoundingBoxes::niBlack_thresholding(const cv::Mat& input, const int& blockSize, const double& k) {
    cv::Mat gray_image, niblack;
    cv::cvtColor(input, gray_image, cv::COLOR_BGR2GRAY);
    cv::ximgproc::niBlackThreshold(gray_image, niblack, 255, cv::THRESH_BINARY, blockSize, k, cv::ximgproc::BINARIZATION_NIBLACK);
    return niblack;
}

cv::Mat BoundingBoxes::saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold) {
    cv::Mat hsv_image, saturation;

    cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
    cv::extractChannel(hsv_image, saturation, 1);

    // Thresholding
    cv::threshold(saturation, saturation, satThreshold, 255, cv::THRESH_BINARY);

    return saturation;
}


cv::Mat minFilter(const cv::Mat& src, const int& kernel_size) {
    // Controls
    if(kernel_size % 2 == 0) throw std::invalid_argument("Error: kernel_size must be odd");
    if(src.channels() != 1) throw std::invalid_argument("Error: the image provided must have only one channel");

    cv::Mat out(src.rows, src.cols, CV_8U);
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            int min = 255;
            int temp;
            for(int x = -kernel_size/2; x <= kernel_size/2; x++) {
                for(int y = -kernel_size/2; y <= kernel_size/2; y++) {
                    if((i+x) >= 0 && (j+y) >= 0 && (i+x) < src.rows && (j+y) < src.cols) { // in case the kernel exceeds the image size
                        temp = src.at<unsigned char> (i+x, j+y);
                        if(temp < min)
                            min = temp;
                    }
                }
            }

            for(int x = -kernel_size/2; x <= kernel_size/2; x++)
                for(int y = -kernel_size/2; y <= kernel_size/2; y++)
                    if((i+x) >= 0 && (j+y) >= 0 && (i+x) < src.rows && (j+y) < src.cols)
                        out.at<unsigned char> (i+x, j+y) = min;
        }
    }
    return out;
}

BoundingBoxes::BoundingBoxes(const cv::Mat &input) {
    int kernelSize = 5;
    int lowThreshold = 100;
    int ratio = 22;
    double GAMMA = 2.5;
    const unsigned int SATURATION_THRESHOLD = 200;
    cv::Mat roiGray, roiCanny;

    // Image Preprocessing
    cv::Mat roiInput = createROI(input);        // Focus on the parking lots, my ROI

    // TODO might consider something with HSV, dont know yet
    cv::Mat image = roiInput.clone();

    // TODO new sequence (GOOD BUT gne)
    cv::Mat gray;
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    int radius = 3;
    int kernel = 2 * radius + 1;
    GaussianBlur(gray, blurred, cv::Size(kernel, kernel), 0);
    // Subtract the blurred image from the original image
    cv::Mat highPass;
    subtract(gray, blurred, highPass);
    highPass = highPass + 128;
    //cv::imshow("high", highPass);
    //cv::waitKey(0);
    cv::Mat otsuThresh;
    threshold(highPass, otsuThresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    //cv::imshow("otsu thresh", otsuThresh);
    //cv::waitKey(0);
    //cv::medianBlur(otsuThresh, otsuThresh, 3);
    //cv::imshow("otsu thresh AFTER", otsuThresh);
    //cv::waitKey(0);


    // TODO THIS sequence is real good
    cv::Mat sugoi;
    cvtColor( roiInput, roiGray, cv::COLOR_BGR2GRAY );
    GaussianBlur(roiGray, roiGray, cv::Size(5,5), 0);
    int blockSize = 5; // Size of the pixel neighborhood used to calculate the threshold
    int C = 2;          // Constant subtracted from the mean or weighted mean
    cv::adaptiveThreshold(roiGray, roiGray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);
    cv::bitwise_not(roiGray, roiGray);
    cv::medianBlur(roiGray, sugoi, 3);
    //cv::imshow("adaptive", sugoi);
    //cv::waitKey(0);


    cv::Mat gc_image = gamma_correction(image, GAMMA);
    //cv::imshow("Gamma", gc_image);
    //cv::waitKey(0);
    cv::Mat saturation = saturation_thresholding(gc_image, SATURATION_THRESHOLD);
    //cv::imshow("Sat", saturation);
    //cv::waitKey(0);
    const unsigned int NIBLACK_BLOCK_SIZE = 7;
    const double NIBLACK_K = 0.6;
    cv::Mat niblack = niBlack_thresholding(gc_image, NIBLACK_BLOCK_SIZE, NIBLACK_K);
    //cv::imshow("NI", niblack);
    //cv::waitKey(0);


    cvtColor( roiInput, roiGray, cv::COLOR_BGR2GRAY );
    GaussianBlur(roiGray, roiGray, cv::Size(kernelSize,kernelSize), 0);
    Canny( roiGray, roiCanny, lowThreshold, lowThreshold*ratio, kernelSize );
    //cv::imshow("Canny", roiCanny);
    //cv::waitKey(0);


    cv::Mat mask =  sugoi | roiCanny | otsuThresh | saturation;
    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    //cv::imshow("After op mask", mask);
    //cv::waitKey(0);
    cv::medianBlur(mask, mask, 3);
    //cv::imshow("After median mask", mask);
    //cv::waitKey(0);
    cv::morphologyEx(mask, mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
    //cv::morphologyEx(mask, mask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    //cv::imshow("After dialte mask", mask);
    //cv::waitKey(0);


    // CONVERSTION TO WHITE BLACK
    cv::bitwise_not(mask, mask);
    cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    mask = createROI(mask);
    cvtColor(mask, mask, cv::COLOR_BGR2GRAY);

    cv::imshow("Final mask", mask);
    cv::waitKey(0);
    mask = minFilter(mask, 7);
    //cv::imshow("After min mask", mask);
    //cv::waitKey(0);
    img = mask;


    /*
    // CORNER DETECTION
    std::vector<cv::Point2f> corners;
    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 0.5;

    cv::Mat postM;
    cv::preCornerDetect(mask, postM, 3);
    cv::imshow("Pre C detect", postM);
    cv::waitKey(0);

    cv::Mat inputMasked = mask & input;
    cvtColor(inputMasked, inputMasked, cv::COLOR_BGR2GRAY);
    cv::imshow("input masked", inputMasked);
    cv::waitKey(0);
    cv::goodFeaturesToTrack(inputMasked, corners, maxCorners, qualityLevel, minDistance);
    // Parameters for corner refinement
    //cv::Size winSize = cv::Size(5, 5);
    //cv::Size zeroZone = cv::Size(-1, -1);
    //cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
    // Refine corners to sub-pixel accuracy
    //cv::cornerSubPix(inputMasked, corners, winSize, zeroZone, criteria);
    // Draw the corners
    cv::Mat corn = input.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        cv::circle(corn, corners[i], 3, cv::Scalar(0, 255, 0), cv::FILLED);
    }
    cv::imshow("Corn", corn);
    cv::waitKey(0);
    */


    /*
    // Hough Transform
    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(mask, hough_lines, 1, CV_PI /180, 40, 25, 4);

    for (auto l : hough_lines) {
        cv::line(hough_lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0, 255));
    }
    */

    // Other hough



    // Close lines that have almost same slope can be removed by using lower angle resolutions for the theta argument of Hough Line method. For example using π/180
    // would result in finding lines that differ only by one degree in their slope. You may use 5*π/180 to find lines in 5 degree resolution.



    // DRAW LINES
    /*
    std::vector<cv::Vec2f> linesDown;
    cv::HoughLines(mask, linesDown, 5, 5 * CV_PI/180, 150, 0, 0, -5* CV_PI / 12, -  7 * CV_PI / 18);
    cv::Mat hough_lines_image = mask.clone();
    cv::cvtColor(mask, hough_lines_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < linesDown.size(); i++) {
        float rho = linesDown[i][0], theta = linesDown[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho ;
        double length = 1500;
        cv::Point pt1(cvRound(x0 + length * (-b)), cvRound(y0 + length * (a)));
        cv::Point pt2(cvRound(x0 - length * (-b)), cvRound(y0 - length * (a)));
        cv::line(hough_lines_image, pt1, pt2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
    }


    /*
    std::vector<cv::Vec2f> linesUp;
    // 55 -- 65 --- 86 the angles
    cv::HoughLines(mask, linesUp, 5, 5 * CV_PI/180, 300, 0, 0,  5* CV_PI / 18,   87* CV_PI / 180);
    cv::Mat hough_lines_image = mask.clone();
    cv::cvtColor(mask, hough_lines_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < linesUp.size(); i++) {
        float rho = linesUp[i][0], theta = linesUp[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho ;
        double length = 1500;
        cv::Point pt1(cvRound(x0 + length * (-b)), cvRound(y0 + length * (a)));
        cv::Point pt2(cvRound(x0 - length * (-b)), cvRound(y0 - length * (a)));
        cv::line(hough_lines_image, pt1, pt2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA); // Blue color for lines2
    }
    */


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat out = input.clone();
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        cv::Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( out, contours, idx, color, cv::FILLED, 8, hierarchy );
    }
    cv::imshow("Test", out);
    cv::waitKey(0);
    /*
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




