/**
 * @author Alessandro Viespoli 2120824
 */
#include "../include/ImageProcessing.hpp"

cv::Mat ImageProcessing::optionalAreaROI(const cv::Size& imgSize) {
    cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);

    // Define ROI that select the third high row of parking spaces
    cv:: RotatedRect roiRect = cv::RotatedRect(cv::Point(1084, 83), cv::Size(452, 54), 28);

    cv::Point2f vertices[4];
    std::vector<cv::Point> contour;
    roiRect.points(vertices);
    for (const cv::Point2f& vertex : vertices) { contour.push_back(vertex); }
    cv::fillConvexPoly(mask, contour, cv::Scalar(255));

    return mask;
}


cv::Mat ImageProcessing::createROI(const cv::Mat& image, const BoundingBox& bBox) {
    // Get the rotated rectangle from the bounding box
    cv::RotatedRect rotatedRect = bBox.getRotatedRect();

    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // Get the bounding rectangle of the rotated rectangle
    cv::Rect boundingRect = rotatedRect.boundingRect(); // ----> cv::Rect and not cv::RotatedRect

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
    cv::flip(warpedImage, warpedImage, 0);  // 0 means flipping around the x-axis

    return warpedImage;
}


cv::Mat ImageProcessing::createRectsMask(const std::vector<cv::RotatedRect>& rotatedRects, const cv::Size& imgSize) {
    cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);

    // For all the given rect, fill the mask with the filled rect
    for (const cv::RotatedRect& rect : rotatedRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        std::vector<cv::Point> verticesVector(4);
        for (unsigned int j = 0; j < 4; ++j)
            verticesVector[j] = vertices[j];
        cv::fillPoly(mask, verticesVector, cv::Scalar(255));
    }

    return mask;
}


cv::Mat ImageProcessing::createMaskDarkColors(const cv::Mat& image) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

    for (unsigned int y = 0; y < image.rows; ++y) {
        for (unsigned int x = 0; x < image.cols; ++x) {
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


cv::Mat ImageProcessing::convertColorMaskToGray(const cv::Mat& segmentationColorMask) {
    cv::Mat classMask = cv::Mat::zeros(segmentationColorMask.size(), CV_8UC1);

    for (unsigned int y = 0; y < segmentationColorMask.rows; ++y) {
        for (unsigned int x = 0; x < segmentationColorMask.cols; ++x) {
            cv::Vec3b color = segmentationColorMask.at<cv::Vec3b>(y, x);

            // Based on the pixel value we create a custom mask for the segmentation
            if (color == cv::Vec3b(0, 0, 0)) {
                classMask.at<uchar>(y, x) = 0; // Class 0: Nothing
            } else if (color == cv::Vec3b(0, 0, 255) || color == cv::Vec3b(1, 1, 1)) {
                classMask.at<uchar>(y, x) = 1; // Class 1: Car inside parking space
            } else if (color == cv::Vec3b(0, 255, 0) || color == cv::Vec3b(2, 2, 2) ) {
                classMask.at<uchar>(y, x) = 2; // Class 2: Car outside parking space
            }
        }
    }

    return classMask;
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

    cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);    // Convert to HSV
    cv::extractChannel(hsv_image, saturation, 1);          // HSV ---> first channel S
    cv::threshold(saturation, saturation, satThreshold, 255, cv::THRESH_BINARY);

    return saturation;
}