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

cv::Mat ParkingLotStatus::createMaskBlackishColors(const cv::Mat& image) const {
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

bool ParkingLotStatus::isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const {
    int whitePixels = cv::countNonZero(mask);
    double whitePixelPercentage = (double)whitePixels / totalPixels * 100.0;
    return whitePixelPercentage >= percentage;
}


void ParkingLotStatus::drawParkingLotStatus() {
    for (BoundingBox& bBox : bBoxes) {
        cv::Point2f vertices[4];
        bBox.getRotatedRect().points(vertices);

        cv::Scalar color;
        if (bBox.isOccupied())
            color = cv::Scalar(0, 0, 255);
        else
            color = cv::Scalar(255, 0, 0);

        for (int j = 0; j < 4; j++)
            cv::line(parkingImage, vertices[j], vertices[(j + 1) % 4], color, 2);

        cv::Point center = bBox.getCenter();
        cv::putText(parkingImage, std::to_string(bBox.getNumber()), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);

    }
}



cv::Mat adjustContrast(const cv::Mat& inputImg, double alpha, int beta) {
    cv::Mat newImage = cv::Mat::zeros(inputImg.size(), inputImg.type());

    // Applica la regolazione di contrasto e luminosità
    inputImg.convertTo(newImage, -1, alpha, beta);

    return newImage;
}

// Variabili globali per i parametri del Canny
int lowThreshold = 100;
int highThreshold = 200;
int kernelSize = 5;

// Funzione di callback per aggiornare l'immagine quando i valori delle trackbar cambiano
void onTrackbarChange(int, void* userdata) {
    // Estrai l'immagine originale dal puntatore passato come userdata
    cv::Mat* originalImage = static_cast<cv::Mat*>(userdata);

    if(kernelSize % 2 == 0)
        kernelSize -= 1;

    // Converti l'immagine in scala di grigi
    cv::Mat gray;
    cv::cvtColor(*originalImage, gray, cv::COLOR_BGR2GRAY);
    // Applica un filtro bilaterale o Gaussian blur
    cv::Mat blurred;
    //gray = adjustContrast(gray, 3, -130);
    cv::GaussianBlur(gray, blurred, cv::Size(kernelSize, kernelSize), 0);

    // Applica l'algoritmo di Canny con i valori di soglia modificati dalle trackbar
    cv::Mat cannyOutput;
    cv::Canny(blurred, cannyOutput, lowThreshold, highThreshold, kernelSize);
    cv::morphologyEx(cannyOutput, cannyOutput, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 2)));

    /// Trova i contorni nell'immagine Canny
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cannyOutput, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Crea una maschera vuota per disegnare i contorni riempiti
    cv::Mat filledContours = cv::Mat::zeros(originalImage->size(), CV_8UC3);

    // Riempie i contorni trovati
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(filledContours, contours, (int)i, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // Mostra l'immagine con i contorni riempiti
    cv::imshow("Canny Output", filledContours);
}









ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes) {

    this->parkingImage = parkingImage;
    for (BoundingBox& bBox : bBoxes) {
        const unsigned int KERNEL_SIZE_CANNY = 5;
        const unsigned int LOW_THRESHOLD = 100;
        const unsigned int RATIO = 22;
        const double GAMMA = 1.25;
        const unsigned int SATURATION_THRESHOLD = 150;
        const unsigned int WHITENESS_THRESHOLD = 150;




        cv::Mat clone = parkingImage.clone();
        cv::Mat boxedInputImg = createROI(parkingImage, bBox);


        // HISTOGRAM BACKPROJECTION

        /*
        // TODO trackbar canny
        // Crea una finestra per visualizzare l'immagine di output
        cv::namedWindow("Canny Output", cv::WINDOW_AUTOSIZE);

        // Crea le trackbar per modificare i parametri di Canny in tempo reale
        cv::createTrackbar("Low Threshold", "Canny Output", &lowThreshold, 5000, onTrackbarChange, &clone);
        cv::createTrackbar("High Threshold", "Canny Output", &highThreshold, 5000, onTrackbarChange, &clone);
        cv::createTrackbar("Kernel Size", "Canny Output", &kernelSize, 10, onTrackbarChange, &clone);

        onTrackbarChange(0, &clone);
        cv::waitKey(0);
        */


        // WHITE CHECK
        int totalPixels = boxedInputImg.rows * boxedInputImg.cols;
        cv::Mat whiteness;
        cv::extractChannel(boxedInputImg, whiteness, 1);
        cv::threshold(whiteness, whiteness, WHITENESS_THRESHOLD, 255, cv::THRESH_BINARY);
        //cv::imshow("white", whiteness);
        //cv::waitKey(0);
        if(isCar(whiteness, totalPixels, 22.0)) {
           // std::cout << "is a car" << std::endl;
            bBox.updateState();
            this->bBoxes.push_back(bBox);
            continue;
        }
        else { // COLOR CHECK
            cv::Mat gc_image = gamma_correction(boxedInputImg, GAMMA);
            cv::Mat saturation = saturation_thresholding(gc_image, SATURATION_THRESHOLD);
            //cv::imshow("sat", saturation);
            //cv::waitKey(0);

            if(isCar(saturation, totalPixels, 15.0)) {
               // std::cout << "is a car" << std::endl;
                bBox.updateState();
                this->bBoxes.push_back(bBox);
                continue;
            }
            else { // BLACK CHECK
                cv::Mat black = createMaskBlackishColors(boxedInputImg);
                //cv::imshow("black", black);
                //cv::waitKey(0);
                if(isCar(black, totalPixels, 10.0)) {
                   // std::cout << "is a car" << std::endl;
                    bBox.updateState();
                    this->bBoxes.push_back(bBox);
                    continue;
                }
                else { // FEATURE BASED
                    cv::Mat gray;
                    cv::Mat des1; // Descriptors
                    std::vector<cv::KeyPoint> kp1; // Keypoints
                    cv::cvtColor(boxedInputImg, gray, cv::COLOR_BGR2GRAY);
                    //gray = adjustContrast(gray, 3, -130);
                    GaussianBlur(gray, gray, cv::Size(3, 3), 0);

                    // Feature
                    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
                    sift->detectAndCompute(gray, cv::noArray(), kp1, des1);
                    cv::Mat imgWithKeypoints;
                    cv::drawKeypoints(boxedInputImg, kp1, imgWithKeypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                   // std::cout << "Number of keypoints detected: " << kp1.size() << std::endl;
                    //cv::imshow("Keypoints", imgWithKeypoints);
                    //cv::waitKey(0);
                    if(kp1.size() >= 30) {
                       // std::cout << "is a car" << std::endl;
                        bBox.updateState();
                        this->bBoxes.push_back(bBox);
                        continue;
                    }
                    else {
                        this->bBoxes.push_back(bBox);
                       // std::cout << "not a car" << std::endl;
                    }
                }

            }
        }

    }


}
