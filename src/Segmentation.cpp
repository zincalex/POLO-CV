/**
 * @author Kabir Bertan 2122545
 */
#include "../include/Segmentation.hpp"

#include <string>
#include <iostream>
#include <opencv2/highgui.hpp>

#include "../include/Graphics.hpp"
#include "../include/ImageProcessing.hpp"

Segmentation::Segmentation(const std::filesystem::path &mogTrainingDir,const std::vector<BoundingBox>& parkingBBoxes,const std::string& imageName) {
    // Parameter loading and definition of needed support matrices
    cv::Mat parkingLab, parkingHSV, parkingGray;
    cv::Mat bgElimMask;
    cv::Mat parkingMasked;
    cv::Mat hsvMask;
    cv::Mat labMask;

    cv::Mat parkingBGR = cv::imread(imageName);
    cv::cvtColor(parkingBGR, parkingLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(parkingBGR, parkingHSV, cv::COLOR_BGR2HSV);
    cv::cvtColor(parkingBGR, parkingGray, cv::COLOR_BGR2GRAY);

    cv::Mat srcClean = parkingBGR.clone(); //used only for final mask application

    // Get the mask of where the parking spaces are located
    cv::Mat parkingSpacesMask = cv::Mat::zeros(parkingGray.size(), CV_8UC1);
    parkingSpacesMask = getBBoxMask(parkingBBoxes, parkingSpacesMask);


    // MOG2 TRAINING on BGR and Lab parking images
    std::vector<cv::String> backgroundImages;
    static bool trained = false;
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog2BGR;
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog2Lab;
    if (!trained) { // training once
        std::cout << "WAIT - Executing segmentation training" << std::endl;
        cv::glob(mogTrainingDir, backgroundImages); // Load images

        // Training
        mog2Lab = trainBackgroundModel(backgroundImages, cv::COLOR_BGR2Lab);
        mog2BGR = trainBackgroundModel(backgroundImages);
        trained = true;
        std::cout << "READY - Training finished" << std::endl;
    }

    // Get saturation values between 75 and 255, full scan on the other channels
    inRange(parkingHSV, cv::Scalar(0, 75, 0), cv::Scalar(179, 255, 255), hsvMask);
    cv::bitwise_not(hsvMask,hsvMask); // invert the mask

    // Pick red cars with higher part of A channel
    inRange(parkingLab, cv::Scalar(0, 140, 0), cv::Scalar(255, 255, 255), labMask);

    // Contours elimination in order to remove noise of little objects
    labMask = contoursElimination(labMask, 300);

    // Create the parkingLab mask by using the trained mog2Lab model
    mog2MaskLab = getForegroundMaskMOG2(mog2Lab, parkingLab);

    // Bitwise AND -----> remove S from mask
    cv::bitwise_and(hsvMask, mog2MaskLab, mog2MaskLab);
    // Bitwise OR -----> we add the red cars that were removed from the last command
    cv::bitwise_or(labMask, mog2MaskLab, mog2MaskLab);

    // Morphological operation to fill little empty holes within the mask
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8)));
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // Find connected components and fill them
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mog2MaskLab, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (unsigned int i = 0; i < contours.size(); i++)
        cv::drawContours(mog2MaskLab, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

    // Contours elimination in order to remove noise of little objects
    mog2MaskLab = contoursElimination(mog2MaskLab, 550);

    // Create the parkingBGR mask by using the trained mog2BGR model
    mog2MaskBGR = getForegroundMaskMOG2(mog2BGR, parkingBGR);

    // Bitwise AND -----> remove S from mask
    cv::bitwise_and(hsvMask, mog2MaskBGR, mog2MaskBGR);
    // Bitwise OR -----> we add the red cars that were removed from the last command
    cv::bitwise_or(labMask, mog2MaskBGR, mog2MaskBGR);

    // Morphological operation to increase the metrics
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    cv::morphologyEx(mog2MaskBGR, mog2MaskBGR, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));

    // We merge the founded mask
    cv::Mat merge;
    cv::bitwise_and(mog2MaskLab, mog2MaskBGR, merge);


    // BACKGROUND ELIMINATION by taking among all the images the one that has the minimum absolute difference
    bgElimMask = backgroundSubtraction(mogTrainingDir, parkingGray);
    cv::bitwise_and(bgElimMask, merge, bgElimMask);
    cv::bitwise_or(bgElimMask, labMask, bgElimMask);
    cv::Mat finalSegmentationMask = bgElimMask.clone();

    // FINAL MASK IMPROVEMENTS
    finalSegmentationMask = contoursElimination(finalSegmentationMask, 1200);  // contours elimination in order to remove noise of little objects
    cv::morphologyEx(finalSegmentationMask, finalSegmentationMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::findContours(finalSegmentationMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (unsigned int i = 0; i < contours.size(); i++)
        cv::drawContours(finalSegmentationMask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

    // Remove cars and noise from the optional area
    cv::Mat blackRoiSegmentation = ImageProcessing::optionalAreaROI(finalSegmentationMask.size());
    cv::bitwise_not(blackRoiSegmentation, blackRoiSegmentation);
    cv::bitwise_and(blackRoiSegmentation, finalSegmentationMask, finalSegmentationMask);


    final_binary_mask = finalSegmentationMask.clone();
    final_mask = getColorMask(finalSegmentationMask, parkingSpacesMask);
    final_image = Graphics::maskApplication(srcClean, final_mask);
}


cv::Mat Segmentation::backgroundSubtraction(const std::filesystem::path &mogTrainingDir, const cv::Mat &parkingLotImg) const {
    cv::Mat bestEmptyImg;
    double minDiff = DBL_MAX;  // Max double value, used to find the minimum

    for (const auto &iter: std::filesystem::directory_iterator(mogTrainingDir)) { // for each image
        std::string imgPath = iter.path().string();
        cv::Mat emptyImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        if (emptyImg.empty()) { // Check for correct loading
            std::cerr << "Error loading image: " << imgPath << std::endl;
            continue;
        }

        // Absdiff between current image and given parking lot image with cars
        cv::Mat diff;
        cv::absdiff(parkingLotImg, emptyImg, diff);

        // Sum up the absolute differences
        double DiffSum = cv::sum(diff)[0];

        // Find and update the best image
        if (DiffSum < minDiff) {
            minDiff = DiffSum;
            bestEmptyImg = emptyImg.clone();
        }
    }

    // Use best match for backgroung elimination
    cv::Mat diff;
    cv::absdiff(parkingLotImg, bestEmptyImg, diff);

    cv::Mat bg1_mask = diff > 0;

    return bg1_mask;
}


cv::Mat Segmentation::contoursElimination(const cv::Mat& inputMask, const int& minArea) const {
    cv::Mat outputMask;
    outputMask = inputMask.clone();

    // Find the connected components
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(outputMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Remove areas less than minArea
    for (unsigned int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < minArea)
            cv::drawContours(outputMask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
    }

    return outputMask;
}


cv::Ptr<cv::BackgroundSubtractorMOG2> Segmentation::trainBackgroundModel(const std::vector<cv::String> &backgroundImages, const int &colorConversionCode) const {
    // Check for mog2 dataset - mandatory for the segmentation, in C++ no possible way of saving an already trained mog2 background remover
    if (backgroundImages.empty()) {
        std::cerr << "Error while loading training dataset, make sure the folder /ParkingLot_dataset/mog2_training_sequence exists and has training images inside" << std::endl;
        exit(1);
    }

    // Create the model
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2(500,14);

    for (const cv::String& imagePath: backgroundImages) {
        cv::Mat image = cv::imread(imagePath);

        if(colorConversionCode != 0)
            cv::cvtColor(image, image, colorConversionCode);

        // Use a automatic learning rate ----> -1
        cv::Mat fgMask;
        mog2->apply(image, fgMask, -1);
    }

    return mog2;
}


cv::Mat Segmentation::getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2>& mog2, const cv::Mat& parkingLotImage) const {
    cv::Mat inputClone = parkingLotImage.clone();

    cv::Mat fgMask;
    mog2->apply(inputClone, fgMask, 0);  // setting lr = 0 stops training

    // Take the pixels classified as probable background (pixel value 127) to 0
    for (unsigned int x = 0; x < fgMask.rows; x++)
        for (unsigned int y = 0; y < fgMask.cols; y++)
            if ((unsigned char) fgMask.at<uchar>(cv::Point(y, x)) == 127)
                fgMask.at<uchar>(cv::Point(y, x)) = 0;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mog2_final_mask = cv::Mat::zeros(fgMask.size(), CV_8UC1);
    for (unsigned int i = 0; i < contours.size(); i++)
        cv::fillPoly(mog2_final_mask, contours[i], cv::Scalar(255, 255, 255));


    return mog2_final_mask;
}


cv::Mat Segmentation::getBBoxMask(const std::vector<BoundingBox> &parkingBBoxes, const cv::Mat& target) const {
    // Exctract from each BoundingBox object its cv::RotatedRect
    std::vector<cv::RotatedRect> extractedRects;
    for (const BoundingBox& bbox : parkingBBoxes)
        extractedRects.push_back(bbox.getRotatedRect());

    return ImageProcessing::createRectsMask(extractedRects, target.size());
}


cv::Mat Segmentation::getColorMask(const cv::Mat& segmentationMask, const cv::Mat& parkingSpacesMask) const {
    const cv::Scalar PARKED_CAR = cv::Scalar(0,0,255);
    const cv::Scalar ROAMING_CAR = cv::Scalar(0,255,0);

    // connected components finds all the connected blobs and labels them
    cv::Mat labels, stats, centroids;
    int totConnected = cv::connectedComponentsWithStats(segmentationMask, labels, stats, centroids); // this method returns the total number of connected components found

    cv::Mat output = cv::Mat::zeros(segmentationMask.size(), CV_8UC3);
    for(unsigned int label = 1; label < totConnected; label++) {
        // load each connected component one at a time ---> look which element in labels has the current label of the for loop
        cv::Mat currentObject = (labels == label);

        // Calculate intersection between parking spaces mask and segmentation mask
        cv::Mat intersect;
        cv::bitwise_and(currentObject, parkingSpacesMask, intersect);

        // If intersection not empty then the pixels corresponding to the current object are assigned as parked car or roaming car
        cv::countNonZero(intersect) > 0 ? output.setTo(PARKED_CAR, currentObject) : output.setTo(ROAMING_CAR, currentObject);
    }

    return output;
}