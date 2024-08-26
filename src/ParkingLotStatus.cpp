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
    std::cout << "Percentage: " << whitePixelPercentage << std::endl;
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

        // Draw a circle at the center of the bounding box with radius 45.0 and color magenta
        cv::circle(parkingImage, center, 45, cv::Scalar(255, 0, 255), 2);  // Magenta color (B, G, R)

        // Draw a yellow point at the center
        cv::circle(parkingImage, center, 3, cv::Scalar(0, 255, 255), -1);  // Yellow color (B, G, R) with filled circle
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



int kernelSizeGaussian = 7; // Size for Gaussian Blur (should be odd)
int blockSize = 9;          // Block size for adaptive thresholding (should be odd)
int C = 3;

// Funzione di callback per aggiornare l'immagine quando i valori delle trackbar cambiano
void onTrackbarChange(int, void* userdata) {
    // Extract the original image from the pointer passed as userdata
    cv::Mat* originalImage = static_cast<cv::Mat*>(userdata);

    // Ensure the kernel size and block size are odd
    if (kernelSizeGaussian % 2 == 0)
        kernelSizeGaussian -= 1;
    if (blockSize % 2 == 0)
        blockSize -= 1;

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(*originalImage, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(kernelSizeGaussian, kernelSizeGaussian), 0);

    // Apply adaptive thresholding
    cv::Mat adaptiveThresholdOutput;
    cv::adaptiveThreshold(blurred, adaptiveThresholdOutput, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, blockSize, C);

    // Find contours in the adaptive thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(adaptiveThresholdOutput, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create an empty mask to draw the filled contours
    cv::Mat filledContours = cv::Mat::zeros(originalImage->size(), CV_8UC3);

    // Fill the contours found
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(filledContours, contours, (int)i, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // Show the image with the filled contours
    cv::imshow("Adaptive Threshold Output", filledContours);
}


void drawAtCenter(cv::RotatedRect rotatedRect, cv::Mat& mask, cv::Mat& image) {
    // Step 1: Calculate the percentage of white pixels in the image
    int whitePixels = cv::countNonZero(mask);  // Count the number of non-zero pixels (white pixels)
    int totalPixels = mask.rows * mask.cols;  // Total number of pixels
    double percentage = (static_cast<double>(whitePixels) / totalPixels) * 100.0;

    // Step 2: Get the center of the rotated rectangle
    cv::Point2f center = rotatedRect.center;

    // Step 3: Convert percentage to a string to display
    std::string percentageText = cv::format("%.2f%%", percentage);

    // Step 4: Choose a font and size for the text
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;

    // Step 5: Calculate the size of the text to center it properly
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(percentageText, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Step 6: Calculate the bottom-left corner of the text to place it centered
    cv::Point textOrigin(center.x - textSize.width / 2, center.y + textSize.height / 2);

    // Step 7: Draw the text on the image
    cv::putText(image, percentageText, textOrigin, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness);
}



ParkingLotStatus::ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes) {

    cv::Mat clone = parkingImage.clone();

    this->parkingImage = parkingImage;
    for (BoundingBox& bBox : bBoxes) {
        this->bBoxes.push_back(bBox);
        const unsigned int KERNEL_SIZE_CANNY = 5;
        const unsigned int LOW_THRESHOLD = 100;
        const unsigned int RATIO = 22;
        const double GAMMA = 1.25;
        const unsigned int SATURATION_THRESHOLD = 150;
        const unsigned int WHITENESS_THRESHOLD = 150;



        cv::Mat boxedInputImg = createROI(parkingImage, bBox);
        //cv::imshow("Box", boxedInputImg);
        //cv::waitKey(0);
        cv::Mat meanShiftResult;
        int spatialRadius = 10;  // Raggio spaziale (spatial window radius)
        int colorRadius = 30;    // Raggio di colore (color window radius)
        //cv::pyrMeanShiftFiltering(clone, meanShiftResult, spatialRadius, colorRadius);
        //cv::imshow("mean", meanShiftResult);
        //cv::waitKey(0);


        cv::Mat gray;
        cv::cvtColor(boxedInputImg, gray, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Mat blurred;
        //cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::bilateralFilter(gray, blurred, 9, 15, 10);
        //cv::imshow("bi",  blurred);
        //cv::waitKey(0);

        // CLAHE
        cv::Mat clahe_image;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(8.0, cv::Size(6, 6));
        clahe->apply(blurred, clahe_image);
        //cv::imshow("clahe",  clahe_image);
        //cv::waitKey(0);

        // Apply adaptive thresholding
        cv::Mat adaptiveThresholdOutput;
        cv::adaptiveThreshold(blurred, adaptiveThresholdOutput, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 9, 3);
        //cv::imshow("PRE med",  adaptiveThresholdOutput);
        //cv::waitKey(0);

        /*
        // Median
        cv::Mat median_blurred_image;
        cv::medianBlur(adaptiveThresholdOutput, median_blurred_image, 5);
        cv::imshow("POST med", median_blurred_image);
        cv::waitKey(0);
        */
        /*
        // Morp
        cv::morphologyEx(median_blurred_image, median_blurred_image, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        cv::imshow("morp", median_blurred_image);
        cv::waitKey(0);
        */


        // Find contours in the adaptive thresholded image
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(adaptiveThresholdOutput, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat filledContours = cv::Mat::zeros(gray.size(), CV_8UC1);
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(filledContours, contours, (int)i, cv::Scalar(255), cv::FILLED);
        }


        cv::imshow("fill", filledContours);
        cv::waitKey(0);
        drawAtCenter(bBox.getRotatedRect(), filledContours, clone);



        /*
        // Suddividi l'immagine nei suoi canali BGR
        std::vector<cv::Mat> bgrChannels(3);
        cv::split(clone, bgrChannels);
        // Crea un'istanza di CLAHE con limiti predefiniti
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));  // Limite di contrasto 2.0, dimensione del tile 8x8
        // Applica CLAHE a ciascun canale (B, G, R)
        for (int i = 0; i < 3; i++) {
            clahe->apply(bgrChannels[i], bgrChannels[i]);
        }
        // Ricostruisci l'immagine equalizzata dai canali BGR
        cv::Mat equalizedImage;
        cv::merge(bgrChannels, equalizedImage);
        // Mostra l'immagine equalizzata
        cv::imshow("Equalized Image", equalizedImage);
        cv::Mat meanShiftResult;
        int spatialRadius = 10;  // Raggio spaziale (spatial window radius)
        int colorRadius = 50;    // Raggio di colore (color window radius)
        cv::pyrMeanShiftFiltering(clone, meanShiftResult, spatialRadius, colorRadius);

        cv::Mat boxedInputImg = createROI(equalizedImage, bBox);
        //cv::Mat boxedInputImg = createROI(parkingImage, bBox);
        const unsigned int KERNEL_SIZE_GAUSSIAN_ADAPTIVE = 5;
        const unsigned int BLOCK_SIZE = 5;                        // Size of the pixel neighborhood used to calculate the threshold
        const unsigned int C = 2;                                 // Constant subtracted from the mean or weighted mean
        cvtColor( parkingImage, clone, cv::COLOR_BGR2GRAY );
        GaussianBlur(clone, clone, cv::Size(KERNEL_SIZE_GAUSSIAN_ADAPTIVE,KERNEL_SIZE_GAUSSIAN_ADAPTIVE), 0);
        cv::adaptiveThreshold(clone, clone, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, BLOCK_SIZE, C);

        std::vector<cv::Vec4i> lines;
        // Crea una maschera per visualizzare le linee trovate
        cv::Mat linesDetected = parkingImage.clone();

        // Crea il rilevatore di segmenti di linee (LSD)
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        // Rileva i segmenti di linea
        lsd->detect(clone, lines);
        cv::Scalar lineColor(0, 0, 255);  // Rosso

        // Spessore delle linee
        int lineThickness = 2;

        // Itera attraverso il vettore di linee e disegna ogni linea
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            cv::line(linesDetected, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), lineColor, lineThickness);
        }


        cv::imshow("before all", parkingImage);
        cv::imshow("mean shift all", linesDetected);
        cv::waitKey(0);
        */

        // HISTOGRAM BACKPROJECTION



        /*
        // TODO trackbar canny
        // Crea una finestra per visualizzare l'immagine di output
        // Create a window to display the output image
            cv::namedWindow("Adaptive Threshold Output", cv::WINDOW_AUTOSIZE);

            // Create trackbars to modify the parameters in real-time
            cv::createTrackbar("Kernel Size Gaussian", "Adaptive Threshold Output", &kernelSizeGaussian, 21, onTrackbarChange, &clone); // Maximum of 21 for Gaussian kernel size
            cv::createTrackbar("Block Size", "Adaptive Threshold Output", &blockSize, 21, onTrackbarChange, &clone); // Maximum of 21 for block size
            cv::createTrackbar("C", "Adaptive Threshold Output", &C, 20, onTrackbarChange, &clone); // Maximum of 20 for C constant

            onTrackbarChange(0, &clone);
            cv::waitKey(0);
        */





        /*
        // REAL METHOD
        // WHITE CHECK
        int totalPixels = boxedInputImg.rows * boxedInputImg.cols;
        cv::Mat whiteness;
        cv::extractChannel(boxedInputImg, whiteness, 1);
        cv::threshold(whiteness, whiteness, WHITENESS_THRESHOLD, 255, cv::THRESH_BINARY);
        cv::imshow("white", whiteness);
        cv::waitKey(0);
        if(isCar(whiteness, totalPixels, 22.0)) {
            std::cout << "is a car" << std::endl;
            bBox.updateState();
            this->bBoxes.push_back(bBox);
            continue;
        }
        else { // COLOR CHECK
            cv::Mat gc_image = gamma_correction(boxedInputImg, GAMMA);
            cv::Mat saturation = saturation_thresholding(gc_image, SATURATION_THRESHOLD);
            cv::imshow("sat", saturation);
            cv::waitKey(0);

            if(isCar(saturation, totalPixels, 15.0)) {
                std::cout << "is a car" << std::endl;
                bBox.updateState();
                this->bBoxes.push_back(bBox);
                continue;
            }
            else { // BLACK CHECK
                cv::Mat black = createMaskBlackishColors(boxedInputImg);
                cv::imshow("black", black);
                cv::waitKey(0);
                if(isCar(black, totalPixels, 10.0)) {
                    std::cout << "is a car" << std::endl;
                    bBox.updateState();
                    this->bBoxes.push_back(bBox);
                    continue;
                }
                else { // FEATURE BASED
                    cv::Mat gray;
                    cv::cvtColor(boxedInputImg, gray, cv::COLOR_BGR2GRAY);

                    // Apply Gaussian blur
                    cv::Mat blurred;
                    cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0);

                    // Apply adaptive thresholding
                    cv::Mat adaptiveThresholdOutput;
                    cv::adaptiveThreshold(blurred, adaptiveThresholdOutput, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 9, 3);

                    // Find contours in the adaptive thresholded image
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(adaptiveThresholdOutput, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    // Create an empty mask to draw the filled contours
                    cv::Mat filledContours = cv::Mat::zeros(gray.size(), CV_8UC1);

                    // Fill the contours found
                    for (size_t i = 0; i < contours.size(); i++) {
                        cv::drawContours(filledContours, contours, (int)i, cv::Scalar(255), cv::FILLED);
                    }

                    cv::imshow("adaptive", filledContours);
                    cv::waitKey(0);
                    if(isCar(filledContours, totalPixels, 27.0)) {
                        std::cout << "is a car" << std::endl;
                        bBox.updateState();
                        this->bBoxes.push_back(bBox);
                        continue;
                    }
                    else {
                        this->bBoxes.push_back(bBox);
                        std::cout << "not a car" << std::endl;
                    }
                }

            }
        }

        */
    }
    cv::imshow("percentage", clone);
    cv::waitKey(0);


}
