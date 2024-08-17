#include "../include/ParkingLotStatus.hpp"

cv::Mat createROI(const cv::Mat& image, const BoundingBox& bBox) {
    // Get the rotated rectangle from the bounding box
    cv::RotatedRect rotatedRect(bBox.getCenter(), bBox.getSize(), bBox.getAngle());

    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // Get the bounding rectangle of the rotated rectangle
    cv::Rect boundingRect = rotatedRect.boundingRect();

    // Define the destination points for perspective transformation
    cv::Point2f dstPoints[4] = {
            cv::Point2f(0, 0),                               // Top-left corner
            cv::Point2f(boundingRect.width - 1, 0),          // Top-right corner
            cv::Point2f(boundingRect.width - 1, boundingRect.height - 1), // Bottom-right corner
            cv::Point2f(0, boundingRect.height - 1)          // Bottom-left corner
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


cv::Mat gamma_correction(const cv::Mat& input, const double& gamma) {
    cv::Mat img_float, img_gamma;

    input.convertTo(img_float, CV_32F, 1.0 / 255.0);    // Convert to float and scale to [0, 1]
    cv::pow(img_float, gamma, img_gamma);               // Gamma correction
    img_gamma.convertTo(img_gamma, CV_8UC3, 255.0);     // Convert back to 8-bit type

    return img_gamma;
}

cv::Mat saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold) {
    cv::Mat hsv_image, saturation;

    cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
    cv::extractChannel(hsv_image, saturation, 1);
    cv::threshold(saturation, saturation, satThreshold, 255, cv::THRESH_BINARY);

    return saturation;
}

cv::Mat createMask(const cv::Mat& image) {
    // Create an empty mask with the same size as the input image and type CV_8U (grayscale)
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

    // Iterate over every pixel in the image
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Get the BGR values at pixel (x, y)
            cv::Vec3b bgrPixel = image.at<cv::Vec3b>(y, x);

            // Check if all BGR components are less than or equal to 30
            if (bgrPixel[0] <= 30 && bgrPixel[1] <= 30 && bgrPixel[2] <= 30) {
                // Set the corresponding mask pixel to white (255)
                mask.at<uchar>(y, x) = 255;
            } else {
                // Set the corresponding mask pixel to black (0) (optional, as mask is initialized to 0)
                mask.at<uchar>(y, x) = 0;
            }
        }
    }

    return mask;
}

bool isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) {
    int whitePixels = cv::countNonZero(mask);
    double whitePixelPercentage = (double)whitePixels / totalPixels * 100.0;

    //std::cout << "---" << std::endl;
    //std::cout << whitePixelPercentage << std::endl;
    return whitePixelPercentage >= percentage;
}


ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes) {

    for (BoundingBox& bBox : bBoxes) {
        // WHITE CHECK
        const double GAMMA = 1.25;
        const unsigned int SATURATION_THRESHOLD = 150;
        const unsigned int WHITENESS_THRESHOLD = 150;

        cv::Mat boxedInputImg = createROI(parkingImage, bBox);
        cv::imshow("SOURCE", boxedInputImg);
        cv::waitKey(0);
        int totalPixels = boxedInputImg.rows * boxedInputImg.cols;
        cv::Mat whiteness;
        cv::extractChannel(boxedInputImg, whiteness, 1);
        cv::threshold(whiteness, whiteness, WHITENESS_THRESHOLD, 255, cv::THRESH_BINARY);
        cv::imshow("white", whiteness);
        cv::waitKey(0);

        if(isCar(whiteness, totalPixels, 22.0)) {
            std::cout << "is a car" << std::endl;
            // aggiorno bbox
            continue;
        }
        else { // COLOR CHECK
            cv::Mat gc_image = gamma_correction(boxedInputImg, GAMMA);
            cv::Mat saturation = saturation_thresholding(gc_image, SATURATION_THRESHOLD);
            cv::imshow("sat", saturation);
            cv::waitKey(0);

            if(isCar(saturation, totalPixels, 15.0)) {
                std::cout << "is a car" << std::endl;
                // aggiorno bbox
                continue;
            }
            else { // BLACK CHECK
                cv::Mat black = createMask(boxedInputImg);
                cv::imshow("black", black);
                cv::waitKey(0);
                if(isCar(black, totalPixels, 10.0)) {
                    std::cout << "is a car" << std::endl;
                    // aggiorno bbox
                    continue;
                }
                else { // NOTHING
                    std::cout << "not a car" << std::endl;

                }
            }

        }





        /*

        cv::cvtColor(gc_image, grayBoxed, cv::COLOR_BGR2GRAY);

        cv::Mat bilatered;
        cv::bilateralFilter(grayBoxed, bilatered, 5, 15, 5);
        Canny(bilatered, final, 300, 300*10, 5);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bilatered, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Crea un'immagine per visualizzare i contorni
        cv::Mat contourImage = cv::Mat::zeros(final.size(), CV_8UC1);


        // Disegna i contorni trovati
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(contourImage, contours, (int)i, cv::Scalar(255, 255, 255), 2, 8, hierarchy, 0);
        }
        */

    }


}
