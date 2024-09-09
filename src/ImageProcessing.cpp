#include "../include/ImageProcessing.hpp"

/*cv::Mat ImageProcessing::createROI(const cv::Mat& input, const bool& obscure) { // We focus the analysis of the image on the parking lots
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    // Define ROI
    std::vector<cv::RotatedRect> rois;
    //rois.push_back(cv::RotatedRect(cv::Point(580, 317), cv::Size(771, 282), 58));
    //rois.push_back(cv::RotatedRect(cv::Point(950, 192), cv::Size(165, 710), 128));
    rois.push_back(cv::RotatedRect(cv::Point(1084, 83), cv::Size(452, 54), 28));

    //black_rois.push_back(cv::RotatedRect(cv::Point(777, 343), cv::Size(1227, 125), 47));
    //black_rois.push_back(cv::RotatedRect(cv::Point(861, 30), cv::Size(1042, 72), 32));

    for (const auto& roiRect : rois) {
        cv::Point2f vertices[4];    // Using cv::Point2f, insted of cv::Point, because it enables the .points method later
        std::vector<cv::Point> contour;

        roiRect.points(vertices);    // Store the vertices of the ROI
        for (auto vertex : vertices) { contour.push_back(vertex); }

        obscure ? cv::fillConvexPoly(mask, contour, cv::Scalar(0)) : cv::fillConvexPoly(mask, contour, cv::Scalar(255));
    }

    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++)
            if (mask.at<uchar>(y, x) == 255)
                result.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(y, x);

    return input;
}*/


cv::Mat ImageProcessing::createRectsMask(const std::vector<cv::RotatedRect>& rotatedRects, const cv::Size& imgSize) {
    cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);
    for (const cv::RotatedRect& rect : rotatedRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        std::vector<cv::Point> verticesVector(4);
        for (unsigned int j = 0; j < 4; j++)
            verticesVector[j] = vertices[j];
        cv::fillPoly(mask, verticesVector, cv::Scalar(255));
    }

    return mask;
}


cv::Mat ImageProcessing::createROI(const cv::Mat& image, const BoundingBox& bBox) {
    // Get the rotated rectangle from the bounding box
    cv::RotatedRect rotatedRect(bBox.getCenter(), bBox.getSize(), bBox.getAngle());

    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // Get the bounding rectangle of the rotated rectangle
    cv::Rect boundingRect = rotatedRect.boundingRect();

    // Define the destination points for perspective transformation
    cv::Point2f dstPoints[4] = {
            cv::Point2f(0, 0),                                              // Top-left corner
            cv::Point2f(boundingRect.width - 1, 0),                         // Top-right corner
            cv::Point2f(boundingRect.width - 1, boundingRect.height - 1),   // Bottom-right corner
            cv::Point2f(0, boundingRect.height - 1)                         // Bottom-left corner
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


cv::Mat ImageProcessing::gamma_correction(const cv::Mat& input, const double& gamma) {
    cv::Mat img_float, img_gamma;

    input.convertTo(img_float, CV_32F, 1.0 / 255.0);    // Convert to float and scale to [0, 1]
    cv::pow(img_float, gamma, img_gamma);               // Gamma correction
    img_gamma.convertTo(img_gamma, CV_8UC3, 255.0);     // Convert back to 8-bit type

    return img_gamma;
}


cv::Mat ImageProcessing::saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold) {
    cv::Mat hsv_image, saturation;

    cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
    cv::extractChannel(hsv_image, saturation, 1);
    cv::threshold(saturation, saturation, satThreshold, 255, cv::THRESH_BINARY);

    return saturation;
}


cv::Mat ImageProcessing::minFilter(const cv::Mat& input, const int& kernel_size) {
    cv::Mat out(input.size(), CV_8U);
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            int min = 255;
            int temp;
            for(int x = -kernel_size/2; x <= kernel_size/2; x++) {
                for(int y = -kernel_size/2; y <= kernel_size/2; y++) {
                    if((i+x) >= 0 && (j+y) >= 0 && (i+x) < input.rows && (j + y) < input.cols) { // in case the kernel exceeds the image size
                        temp = input.at<unsigned char> (i + x, j + y);
                        if(temp < min)
                            min = temp;
                    }
                }
            }

            for(int x = -kernel_size/2; x <= kernel_size/2; x++)
                for(int y = -kernel_size/2; y <= kernel_size/2; y++)
                    if((i+x) >= 0 && (j+y) >= 0 && (i+x) < input.rows && (j + y) < input.cols)
                        out.at<unsigned char> (i+x, j+y) = min;
        }
    }
    return out;
}


cv::Mat ImageProcessing::adjustContrast(const cv::Mat& inputImg, const double& contrastFactor, const int& brightnessOffset) {
    cv::Mat newImage = cv::Mat::zeros(inputImg.size(), inputImg.type());

    // Contrast and lighting regulation
    inputImg.convertTo(newImage, -1, contrastFactor, brightnessOffset);
    return newImage;
}


cv::Mat ImageProcessing::morphologicalSkeleton(const cv::Mat& binaryImg) {
    cv::Mat skeleton = cv::Mat::zeros(binaryImg.size(), CV_8UC1);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::Mat img = binaryImg.clone();

    cv::Mat temp, eroded;
    while (true) {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skeleton, temp, skeleton);

        img = eroded.clone();

        // If nothing left to erode, stop the cycle
        if (cv::countNonZero(img) == 0)
            break;
    }

    return skeleton;
}


cv::Mat ImageProcessing::applyCLAHE(const cv::Mat& input){
    cv::Mat CLAHEimage;
    cv::cvtColor(input, CLAHEimage, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(CLAHEimage, lab_planes);

    // Apply CLAHE to the L (lightness) channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0); // Set clip limit
    clahe->setTilesGridSize(cv::Size(8, 8)); // Set tile grid size

    cv::Mat clahe_l;
    clahe->apply(lab_planes[0], clahe_l); // Apply CLAHE to the L channel

    // Merge the modified L channel back with the original A and B channels
    lab_planes[0] = clahe_l;
    cv::Mat lab_clahe_image;
    cv::merge(lab_planes, lab_clahe_image);

    // Convert the LAB image back to BGR color space
    cv::Mat clahe_bgr_image;
    cv::cvtColor(lab_clahe_image, clahe_bgr_image, cv::COLOR_Lab2BGR);
    return clahe_bgr_image;
}


cv::Mat ImageProcessing::createMaskDarkColors(const cv::Mat& image) {
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