#include "../include/ParkingSpaceDetector.hpp"

cv::Mat ParkingSpaceDetector::createROI(const cv::Mat& input) { // We focus the analysis of the image on the parking lots
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    // Define ROIs
    std::vector<cv::RotatedRect> rois;
    rois.push_back(cv::RotatedRect(cv::Point(580, 317), cv::Size(771, 282), 58));
    rois.push_back(cv::RotatedRect(cv::Point(950, 192), cv::Size(165, 710), 128));
    rois.push_back(cv::RotatedRect(cv::Point(1084, 83), cv::Size(452, 54), 28));

    // More ROI in order to refine the ROI selected
    std::vector<cv::RotatedRect> black_rois;
    black_rois.push_back(cv::RotatedRect(cv::Point(777, 343), cv::Size(1227, 125), 47));
    black_rois.push_back(cv::RotatedRect(cv::Point(861, 30), cv::Size(1042, 72), 32));

    for (const auto& roiRect : rois) {
        cv::Point2f vertices[4];    // Using cv::Point2f, insted of cv::Point, because it enables the .points method later
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

cv::Mat ParkingSpaceDetector::gamma_correction(const cv::Mat& input, const double& gamma) {
    cv::Mat img_float, img_gamma;

    input.convertTo(img_float, CV_32F, 1.0 / 255.0);    // Convert to float and scale to [0, 1]
    cv::pow(img_float, gamma, img_gamma);               // Gamma correction
    img_gamma.convertTo(img_gamma, CV_8UC3, 255.0);     // Convert back to 8-bit type

    return img_gamma;
}

cv::Mat ParkingSpaceDetector::saturation_thresholding(const cv::Mat& input, const unsigned int& satThreshold) {
    cv::Mat hsv_image, saturation;

    cv::cvtColor(input, hsv_image, cv::COLOR_BGR2HSV);
    cv::extractChannel(hsv_image, saturation, 1);
    cv::threshold(saturation, saturation, satThreshold, 255, cv::THRESH_BINARY);

    return saturation;
}

cv::Mat ParkingSpaceDetector::minFilter(const cv::Mat& input, const int& kernel_size) {
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

cv::Mat ParkingSpaceDetector::maskCreation(const cv::Mat& inputImg) {
    const int KERNEL_SIZE_GAUSSIAN_OTSU = 9;

    const unsigned int KERNEL_SIZE_GAUSSIAN_ADAPTIVE = 5;
    const unsigned int BLOCK_SIZE = 5;                        // Size of the pixel neighborhood used to calculate the threshold
    const unsigned int C = 2;                                 // Constant subtracted from the mean or weighted mean
    const unsigned int KERNEL_SIZE_MEDIAN_ADAPTIVE = 3;

    const double GAMMA = 2.5;
    const unsigned int SATURATION_THRESHOLD = 150;

    const unsigned int KERNEL_SIZE_CANNY = 5;
    const unsigned int LOW_THRESHOLD = 100;
    const unsigned int RATIO = 22;

    const unsigned int KERNEL_SIZE_CLOSING = 3;
    const unsigned int KERNEL_SIZE_MIN = 5;

    // TODO in case he want something general we create the mask then we do a template matching with a rect that increases size
    // and that rotate with bin of 5 degree from 90 to -90, if the white part of the image correspond with lets say > 60 of the rect than is match
    // in that case the final mask need to change. We do it that if sat mask has 1 the 0 in the final otherwise we keep what there is in the union of the others

    // Focus the masking, consider only the 3 main areas where the parking lots are
    cv::Mat roiInput = createROI(inputImg);

    // Otsu mask
    cv::Mat gray, blurred, highPass, otsuThresh;
    cvtColor(roiInput, gray, cv::COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, cv::Size(KERNEL_SIZE_GAUSSIAN_OTSU, KERNEL_SIZE_GAUSSIAN_OTSU), 0);
    subtract(gray, blurred, highPass);  // Subtract the blurred image from the original image
    highPass = highPass + 128;
    threshold(highPass, otsuThresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    // Adaptive mask
    cv::Mat adaptive, roiGray;
    cvtColor( roiInput, roiGray, cv::COLOR_BGR2GRAY );
    GaussianBlur(roiGray, roiGray, cv::Size(KERNEL_SIZE_GAUSSIAN_ADAPTIVE,KERNEL_SIZE_GAUSSIAN_ADAPTIVE), 0);
    cv::adaptiveThreshold(roiGray, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, BLOCK_SIZE, C);
    cv::bitwise_not(adaptive, adaptive);
    cv::medianBlur(adaptive, adaptive, KERNEL_SIZE_MEDIAN_ADAPTIVE);

    // Saturation mask
    cv::Mat gc_image = gamma_correction(roiInput, GAMMA);
    cv::Mat saturation = saturation_thresholding(gc_image, SATURATION_THRESHOLD);

    // Canny mask
    cv::Mat roiCanny;
    cvtColor( roiInput, roiGray, cv::COLOR_BGR2GRAY );
    GaussianBlur(roiGray, roiGray, cv::Size(KERNEL_SIZE_CANNY, KERNEL_SIZE_CANNY), 0);
    Canny(roiGray, roiCanny, LOW_THRESHOLD, LOW_THRESHOLD * RATIO, KERNEL_SIZE_CANNY );

    // Union of the masks
    cv::Mat mask = adaptive | roiCanny | otsuThresh | saturation;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(KERNEL_SIZE_CLOSING, KERNEL_SIZE_CLOSING)));
    cv::bitwise_not(mask, mask);                      // Interested to find the areas between the lines,
    cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    mask = createROI(mask);                              // Adjust the white areas outside the ROI
    cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    mask = minFilter(mask, KERNEL_SIZE_MIN);

    return mask;
}

bool ParkingSpaceDetector::isWithinRadius(const std::pair<int, int>& center, const std::pair<int, int>& point, const double& radius) {
    double distance = std::sqrt(std::pow(center.first - point.first, 2) + std::pow(center.second - point.second, 2));
    return distance <= radius;
}

std::map<std::pair<int, int>, cv::Rect> ParkingSpaceDetector::nonMaximaSuppression(const std::map<std::pair<int, int>, cv::Rect>& parkingLotsBoxes, const float& iouThreshold) {
    if (parkingLotsBoxes.size() == 1) return {parkingLotsBoxes}; // Only one candidate, hence my only bounding box

    std::vector<cv::Rect> rects;
    std::vector<std::pair<int, int>> centers;
    std::vector<int> indices;
    std::map<std::pair<int, int>, cv::Rect> validCandidates;

    for (const auto& entry : parkingLotsBoxes) {   // entry = (center, rect)
        centers.push_back(entry.first);
        rects.push_back(entry.second);
    }

    // Despite being inside the deep neural network library, the function does NOT use deep learning
    cv::dnn::NMSBoxes(rects, std::vector<float>(rects.size(), 1.0f), 0.0f, iouThreshold, indices);

    // Populate the map
    for (int idx : indices)
        validCandidates[centers[idx]] = rects[idx];
    return validCandidates;
}

std::vector<std::pair<cv::Point, cv::Rect>> ParkingSpaceDetector::computeAverageRect(const std::vector<std::map<std::pair<int, int>, cv::Rect>>& boundingBoxesNMS) {
    std::vector<std::pair<cv::Point, cv::Rect>> averages;

    for (const auto& parkingSpace : boundingBoxesNMS) {
        unsigned int sumXCenter = 0, sumYCenter = 0, sumXTopLeft = 0, sumYTopLeft = 0;
        unsigned int sumWidth = 0, sumHeight = 0;
        unsigned int count = parkingSpace.size();

        for (const auto& box : parkingSpace) {
            sumXCenter += box.first.first;
            sumYCenter += box.first.second;
            sumXTopLeft += box.second.x;
            sumYTopLeft += box.second.y;
            sumWidth += box.second.width;
            sumHeight += box.second.height;
        }

        cv::Point avgCenter(static_cast<int>(sumXCenter / count), static_cast<int>(sumYCenter / count));
        cv::Rect avgRect = cv::Rect(static_cast<int>(sumXTopLeft / count), static_cast<int>(sumYTopLeft / count),
                                    static_cast<int>(sumWidth / count), static_cast<int>(sumHeight / count));
        averages.push_back(std::make_pair(avgCenter, avgRect));
    }
    return averages;
}

std::vector<cv::RotatedRect> ParkingSpaceDetector::rotateBoundingBoxes(const std::vector<std::pair<cv::Point, cv::Rect>>& rects, const float& angle) {
    std::vector<cv::RotatedRect> rotatedBBoxes;
    for (const auto& pair : rects) {
        cv::Point center = pair.first;
        cv::Rect rect = pair.second;

        cv::Size size(rect.width, rect.height);
        cv::RotatedRect rotatedBBox(center, size, angle);
        rotatedBBoxes.push_back(rotatedBBox);
    }
    return rotatedBBoxes;
}


// TODO create class for image pre processing
ParkingSpaceDetector::ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir) {

    const double RADIUS = 30.0;
    const float IOU_THRESHOLD = 0.95;
    const float ANGLE = 10.0;

    std::map<std::pair<int, int>, cv::Rect> boundingBoxesCandidates;

    // Image preprocessing and find the candidate
    for (const auto& iter : std::filesystem::directory_iterator(emptyFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat input = cv::imread(imgPath);
        if (input.empty()) {
            std::cerr << "Error opening the image" << std::endl;
        }

        cv::Mat mask = maskCreation(input);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        /*
         * DRAW THE MASK COLORED
        cv::Mat out = input.clone();
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            cv::Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours( out, contours, idx, color, cv::FILLED, 8, hierarchy );
        }
        cv::imshow("Test", out);
        cv::waitKey(0);
        */

        // Save the information for bounding box candidates
        for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
            cv::Rect rect = cv::boundingRect(contours[idx]);

            // Filter out small and large rects
            if (rect.height > 18 && rect.width > 18 && rect.height < 200 && rect.width < 200) {
                cv::Point center = (rect.tl() + rect.br()) * 0.5;

                // The key cv::Point cannot be used because it has not the operator< , which maps relies on
                boundingBoxesCandidates.insert({std::make_pair(center.x, center.y), rect});
            }
        }
    }

    std::vector<std::map<std::pair<int, int>, cv::Rect>> boundingBoxesNonMaximaSupp;
    while (!boundingBoxesCandidates.empty()) {
        std::map<std::pair<int, int>, cv::Rect> parkingSpaceBoxes;

        // First populate the map with the first not analyzed parking space
        auto iterator = boundingBoxesCandidates.begin();
        std::pair<int, int> centerParkingSpace = iterator->first;
        parkingSpaceBoxes[centerParkingSpace] = iterator ->second;
        boundingBoxesCandidates.erase(iterator); // remove it in order to not insert it twice

        // Look for all the other candidates if there is one that represent the same parking lot
        auto iterator2 = boundingBoxesCandidates.begin();
        while (iterator2 != boundingBoxesCandidates.end()) {
            std::pair<int, int> anotherCenter = iterator2->first;
            if (isWithinRadius(centerParkingSpace, anotherCenter, RADIUS)) {
                parkingSpaceBoxes[anotherCenter] = iterator2->second;
                iterator2 = boundingBoxesCandidates.erase(iterator2);  // Erase and get the next iterator
            } else {
                ++iterator2;  // Pre-increment for efficiency purpose
            }
        }

        // All candidates for a parking space are found, need to clear them with nms
        std::map<std::pair<int, int>, cv::Rect> validBoxes = nonMaximaSuppression(parkingSpaceBoxes, IOU_THRESHOLD);
        boundingBoxesNonMaximaSupp.push_back(validBoxes);
    }

    std::vector<std::pair<cv::Point, cv::Rect>> finalBoundingBoxes = computeAverageRect(boundingBoxesNonMaximaSupp);
    std::vector<cv::RotatedRect> finalRotatedBoundingBoxes = rotateBoundingBoxes(finalBoundingBoxes, ANGLE);

    unsigned short parkNumber = 1;
    for (const cv::RotatedRect rotRect : finalRotatedBoundingBoxes) {
        BoundingBox bbox = BoundingBox(rotRect, parkNumber++);
        bBoxes.push_back(bbox);
    }
}