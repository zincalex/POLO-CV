//
// Created by trigger on 9/6/24.
//

#include "../include/Segmentation.hpp"


cv::Mat Segmentation::averageEmptyImages(const std::filesystem::path &emptyFramesDir) {
    std::vector<cv::Mat> empty_images;
    for (const auto &iter: std::filesystem::directory_iterator(emptyFramesDir)) {
        std::string imgPath = iter.path().string();
        cv::Mat input = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        empty_images.push_back(input);
    }//loads all images from sequence 0 in vector
    cv::Mat empty_parking = cv::Mat::zeros(empty_images[0].size(), CV_32FC1);
    for (const auto &img: empty_images) {
        cv::Mat temp;
        img.convertTo(temp, CV_32FC1);
        empty_parking += temp;
    }//sum all the images in a float matrix so the mean can be more precise
    empty_parking /= empty_images.size();
    //calculate mean

    empty_parking.convertTo(empty_parking, CV_8UC1);
    return empty_parking;
}

cv::Mat Segmentation::backgroundSubtractionMask(const cv::Mat &empty_parking, const cv::Mat &busy_parking) {
    cv::Mat diff;
    cv::absdiff(busy_parking, empty_parking, diff);
    cv::Mat bg1_mask;
    cv::threshold(diff, bg1_mask, 50, 255, cv::THRESH_BINARY);
    //use absolute difference and tresholding to do background subtraction

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_OPEN, kernel);


    return bg1_mask;
}

cv::Mat Segmentation::smallContoursElimination(const cv::Mat& input_mask, const int&minArea) {
    cv::Mat in_mask;
    in_mask = input_mask.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(in_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// filter contours by size deleting the ones smaller than the area defined in the method call
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < minArea) {
            cv::drawContours(in_mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }
    return in_mask;
}

cv::Ptr<cv::BackgroundSubtractorMOG2> Segmentation::trainBackgroundModel(const std::vector<cv::String>& backgroundImages) {
    //check for mog2 dataset
    if (backgroundImages.empty()) {
        std::cerr << "Error while loading training dataset, make sure the folder /ParkingLot_dataset/mog2_training_sequence exists and has training images inside" << std::endl;
        exit(1);
    }

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2();

    for (const auto &imagePath: backgroundImages) {
        cv::Mat image = cv::imread(imagePath);

        // using apply with a learning rate different from 0 executes the learning
        cv::Mat fgMask;
        mog2->apply(image, fgMask, 0.55);
    }
    return mog2;
}

cv::Mat Segmentation::getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog2, cv::Mat &busy_parking) {
    cv::Mat image_src = busy_parking.clone();
    cv::Mat fgMask;
    mog2->apply(image_src, fgMask, 0);  // setting lr = 0 stops training
    fgMask = smallContoursElimination(fgMask,100);

    cv::waitKey(0);

    for (int x = 0; x < fgMask.rows; x++) {
        for (int y = 0; y < fgMask.cols; y++) {
            if ((unsigned char) fgMask.at<uchar>(cv::Point(y, x)) == 127) {
                fgMask.at<uchar>(cv::Point(y, x)) = 0;
            }
        }
    }

    fgMask = smallContoursElimination(fgMask,100);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat drawing = cv::Mat::zeros(fgMask.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        cv::fillPoly(drawing, contours[i], cv::Scalar(255, 255, 255)); // Riempi con colore verde
    }
    //draws filled polygons from the mask to fill some gaps in the mask
    return drawing;
}

cv::Mat Segmentation::getBBoxMask(const std::vector<BoundingBox> &parkingBBoxes, cv::Mat& target) {
    //loads the bounding boxes in a vector and uses create rect mask function to create the mask defining the "parked car" class
    std::vector<cv::RotatedRect> extractedRects;
    for (const auto& bbox : parkingBBoxes) {
        extractedRects.push_back(bbox.getRotatedRect());
    }
    cv::Mat rectsMask = ImageProcessing::createRectsMask(extractedRects, target.size());
    return rectsMask;
}

cv::Mat Segmentation::getColorMask(const cv::Mat &car_fgMask, const cv::Mat & parking_mask) {

    const cv::Scalar PARKED_CAR = cv::Scalar(0,0,255);
    const cv::Scalar ROAMING_CAR = cv::Scalar(0,255,0);

    //connected components finds all the connected blobs and labels them
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(car_fgMask, labels,stats, centroids);

    cv::Mat out_mask = cv::Mat::zeros(car_fgMask.size(), CV_8UC3);

    for(int label = 1; label < num_labels; label++){
        //load the blobs one at the time
        cv::Mat current_object = (labels == label);

        //calculate intersection between parking spaces mask and segmentation mask
        cv::Mat intersect;
        cv::bitwise_and(current_object, parking_mask, intersect);
        //if intersection not empty then the pixels corresponding to the blob are assigned as parked car or roaming car
        if(cv::countNonZero(intersect)>0){
            out_mask.setTo(PARKED_CAR, current_object);
        } else{
            out_mask.setTo(ROAMING_CAR, current_object);
        }
    }
    return out_mask;
}

//simple public methods to return the results and use them outside the class
cv::Mat Segmentation::getSegmentationResult() {
    return final_image;
}

cv::Mat Segmentation::getSegmentationMaskWithClasses() {
    return final_mask;
}

cv::Mat Segmentation::getSegmentationMaskBinary() {
    return final_binary_mask;
}

Segmentation::Segmentation(const std::filesystem::path &emptyFramesDir, const std::filesystem::path &mogTrainingDir,const std::vector<BoundingBox>& parkingBBoxes,const std::string& imageName) {
        //parameter loading and definition of needed support matrices
        cv::Mat parking_with_cars_col = cv::imread(imageName);
        cv::Mat parking_with_cars;
        cv::cvtColor(parking_with_cars_col, parking_with_cars, cv::COLOR_BGR2GRAY);
        cv::Mat src_clean = parking_with_cars_col.clone(); //used only for final mask application
        cv::Mat mask;
        cv::Mat empty_parking = averageEmptyImages(emptyFramesDir);
        cv::Mat parking_masked;
        std::vector<cv::String> backgroundImages;
        cv::Mat parkingSpaceMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
        parkingSpaceMask = getBBoxMask(parkingBBoxes, parkingSpaceMask);

        //load images and execute MOG2 training and application
        cv::glob(mogTrainingDir, backgroundImages);
        cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = trainBackgroundModel(backgroundImages);
        cv::Mat bgSubctMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
        cv::Mat Mog2Mask = getForegroundMaskMOG2(mog2, parking_with_cars_col);

        //execute background subtraction and use bw& to merge the masks making the mog2 more robust to illumination change
        mask = backgroundSubtractionMask(empty_parking, parking_with_cars);
        cv::bitwise_and(mask, Mog2Mask, bgSubctMask);


        //morphological to clean the final masks as best as possible
        cv::morphologyEx(bgSubctMask, bgSubctMask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 16)));
        cv::morphologyEx(bgSubctMask, bgSubctMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

        //removes the small noise
        bgSubctMask = smallContoursElimination(bgSubctMask, 1500);

        //assign variables for access
        final_binary_mask = bgSubctMask.clone();
        final_mask = getColorMask(bgSubctMask, parkingSpaceMask);
        final_image = Graphics::maskApplication(src_clean, final_mask);

}







