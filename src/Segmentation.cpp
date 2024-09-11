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
    }//sum all the images in a float matrix so the mean can be more precise and is not bound by unsigned char
    empty_parking /= empty_images.size();
    //calculate mean

    empty_parking.convertTo(empty_parking, CV_8UC1);
    return empty_parking;
}

cv::Mat Segmentation::backgroundSubtractionMask(const cv::Mat &empty_parking, const cv::Mat &busy_parking) {
    cv::Mat diff;
    cv::absdiff(busy_parking, empty_parking, diff); //used absdiff to avoid possible negative numbers
    cv::Mat bg1_mask;
    cv::threshold(diff, bg1_mask, 60, 255, cv::THRESH_BINARY);
    //use tresholding to get the mask

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
    cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_DILATE, kernel);
    //cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_CLOSE, kernel);
    //use a huge morphological operation, in this case this mask only removes a bigger portion of the background to help mog2
    //cv::imshow("bgelim", bg1_mask);
    return bg1_mask;
}

cv::Mat Segmentation::smallContoursElimination(const cv::Mat& input_mask, const int&minArea) {
    cv::Mat in_mask;
    in_mask = input_mask.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(in_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // filter contours by size deleting the ones smaller than the area defined in the method call, reduces noise in the mask without using morphologicals
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < minArea) {
            cv::drawContours(in_mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }
    return in_mask;
}

cv::Ptr<cv::BackgroundSubtractorMOG2> Segmentation::trainBackgroundModel(const std::vector<cv::String> &backgroundImages, const int &color_conversion_code) {
    //check for mog2 dataset - mandatory for the segmentation, in c++ no possibile way of saving an already trained mog2 bg remover
    if (backgroundImages.empty()) {
        std::cerr << "Error while loading training dataset, make sure the folder /ParkingLot_dataset/mog2_training_sequence exists and has training images inside" << std::endl;
        exit(1);
    }

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2(500,14);

    for (const auto &imagePath: backgroundImages) {
        cv::Mat image = cv::imread(imagePath);
        if(color_conversion_code != 0)
            cv::cvtColor(image, image, color_conversion_code);

        // using apply with a learning rate different from 0 executes the training, -1 uses automatic lr
        cv::Mat fgMask;
        mog2->apply(image, fgMask, -1);
    }
    return mog2;
}

cv::Mat Segmentation::getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog2, cv::Mat &busy_parking) {
    cv::Mat image_src = busy_parking.clone();
    cv::Mat fgMask;
    mog2->apply(image_src, fgMask, 0);  // setting lr = 0 stops training


    //shadows and possible foreground are labeled with px =127, remove only keeping sure fg
    for (int x = 0; x < fgMask.rows; x++) {
        for (int y = 0; y < fgMask.cols; y++) {
            if ((unsigned char) fgMask.at<uchar>(cv::Point(y, x)) == 127) {
                fgMask.at<uchar>(cv::Point(y, x)) = 0;
            }
        }
    }

    //fgMask = smallContoursElimination(fgMask,100);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mog2_final_mask = cv::Mat::zeros(fgMask.size(), CV_8UC1);



    for (size_t i = 0; i < contours.size(); i++) {
        cv::fillPoly(mog2_final_mask, contours[i], cv::Scalar(255, 255, 255));
    }
    //draws filled polygons from the mask to fill some gaps in the mask
    //cv::morphologyEx(mog2_final_mask, mog2_final_mask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    //cv::imshow("TEST", mog2_final_mask);
    return mog2_final_mask;
}

cv::Mat Segmentation::getBBoxMask(const std::vector<BoundingBox> &parkingBBoxes, cv::Mat& target) {
    //loads the bounding boxes in a vector and uses create rect mask function to create the mask defining the "parked car" class
    std::vector<cv::RotatedRect> extractedRects;
    for (const auto& bbox : parkingBBoxes) {
        extractedRects.push_back(bbox.getRotatedRect());
    }
    //classifies the optional area as parking spaces since it is ignored in the parkingspacedetector as per project specifications
    cv::Mat optionalArea = ImageProcessing::optionalAreaROI(target.size());
    cv::Mat rectsMask = ImageProcessing::createRectsMask(extractedRects, target.size());

    //returns merged masks with three white ROIs where the parking slots are
    return optionalArea | rectsMask;
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

cv::Mat Segmentation::getSegmentationResult() {
    return final_image;
}

cv::Mat Segmentation::getSegmentationMaskWithClasses() {
    return final_mask;
}


cv::Mat Segmentation::getSegmentationMaskBinary() {
    return final_binary_mask;
}

cv::Mat Segmentation::getMOG2Labmask() {
    return mog2MaskLab;
}


int Segmentation::dynamicContoursThresh(const cv::Mat &mask_to_filter) {
    int num_mask_pixels = cv::countNonZero(mask_to_filter);
    int threshold;
    if (num_mask_pixels < 50000){
        threshold = 1500;
    } else {
        threshold = 300;
    }
    return  threshold;
}



Segmentation::Segmentation(const std::filesystem::path &emptyFramesDir, const std::filesystem::path &mogTrainingDir,const std::vector<BoundingBox>& parkingBBoxes,const std::string& imageName) {
    //parameter loading and definition of needed support matrices
    cv::Mat parking_with_cars_col = cv::imread(imageName);
    cv::Mat parking_Lab;
    cv::Mat parking_HSV;
    cv::Mat parking_with_cars;
    cv::cvtColor(parking_with_cars_col, parking_Lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(parking_with_cars_col, parking_HSV, cv::COLOR_BGR2HSV);
    cv::cvtColor(parking_with_cars_col, parking_with_cars, cv::COLOR_BGR2GRAY);
    cv::Mat src_clean = parking_with_cars_col.clone(); //used only for final mask application
    cv::Mat bgElimMask;
    cv::Mat parking_masked;


    cv::Mat hsvMask;
    cv::Mat labMask;


    std::vector<cv::String> backgroundImages;
    cv::Mat parkingSpaceMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
    cv::Mat bgSubctMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
    parkingSpaceMask = getBBoxMask(parkingBBoxes, parkingSpaceMask);

    static bool trained = false;
    static cv::Mat empty_parking;
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog2bgr;
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog2hsv;

    if (!trained) {
        std::cout << "WAIT - Executing training, action will be ran only once" << std::endl;
        //load images and execute MOG2 training and application
        cv::glob(mogTrainingDir, backgroundImages);
        mog2hsv = trainBackgroundModel(backgroundImages, cv::COLOR_BGR2Lab);
        mog2bgr = trainBackgroundModel(backgroundImages);
        empty_parking = averageEmptyImages(mogTrainingDir);
        trained = true;
        std::cout << "READY - Training finished" << std::endl;
    }


    inRange(parking_HSV, cv::Scalar(0, 75, 0), cv::Scalar(179, 255, 255), hsvMask);
    inRange(parking_Lab, cv::Scalar(0, 140, 0), cv::Scalar(255, 255, 255), labMask);
    labMask = smallContoursElimination(labMask, 300);
    cv::bitwise_not(hsvMask,hsvMask);
    //cv::imshow("HSVMASK", hsvMask);
    //cv::imshow("LABMASK", labMask);
    mog2MaskLab = getForegroundMaskMOG2(mog2hsv, parking_Lab);
    //cv::imshow("LAB_PRE", mog2MaskLab);
    cv::bitwise_and(hsvMask, mog2MaskLab, mog2MaskLab);
    cv::bitwise_or(labMask, mog2MaskLab, mog2MaskLab);
    cv::imshow("LAB_POST", mog2MaskLab);
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8)));
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    //cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 1)));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mog2MaskLab, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // filter contours by size deleting the ones smaller than the area defined in the method call, reduces noise in the mask without using morphologicals
    for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(mog2MaskLab, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
    }

    mog2MaskLab = smallContoursElimination(mog2MaskLab, 550);
    cv::imshow("LAB_POSTPOSTPOST", mog2MaskLab);
    cv::Mat mog2MaskBGR = getForegroundMaskMOG2(mog2bgr, parking_with_cars_col);
    cv::bitwise_and(hsvMask, mog2MaskBGR, mog2MaskBGR);
    cv::bitwise_or(labMask, mog2MaskBGR, mog2MaskBGR);
    //cv::imshow("BGR_PRE", mog2MaskBGR);
    //mog2MaskBGR = smallContoursElimination(mog2MaskBGR, 50);
    cv::imshow("BGR_Post", mog2MaskBGR);
    //mog2MaskBGR = smallContoursElimination(mog2MaskBGR, 300);
    cv::morphologyEx(mog2MaskLab, mog2MaskLab , cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    cv::morphologyEx(mog2MaskBGR, mog2MaskBGR, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    cv::imshow("Final_BGR", mog2MaskBGR);
    cv::imshow("Final_LAB", mog2MaskLab);

    //cv::morphologyEx(mog2MaskBGR, mog2MaskBGR, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    //cv::morphologyEx(mog2MaskLab, mog2MaskLab, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 6)));
    cv::Mat merge;
    cv::bitwise_and(mog2MaskLab, mog2MaskBGR, merge);
    mog2MaskBGR = smallContoursElimination(mog2MaskBGR, 300);
    //cv::imshow("BGR", mog2MaskBGR);
    //cv::imshow("Lab", mog2MaskLab);
    //Mog2Mask = smallContoursElimination(bgSubctMask, 100);

    //execute background subtraction and use bw& to merge the masks making the mog2 more robust to illumination change
    bgElimMask = backgroundSubtractionMask(empty_parking, parking_with_cars);
    //cv::imshow("bgelim", bgElimMask);
    //cv::imshow("mog2", Mog2Mask);
    //cv::bitwise_and(bgElimMask, merge, bgSubctMask);
    //bgSubctMask = Mog2Mask.clone();

    //final mask refinement
    //morphological to clean the final masks as best as possible
    //cv::morphologyEx(bgSubctMask, bgSubctMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4)));

    //removes the small noise
    int small_contour = dynamicContoursThresh(bgSubctMask);
    merge = smallContoursElimination(merge, 1200);
    //std::cout << cv::countNonZero(bgSubctMask) << std::endl; //300k+
    cv::morphologyEx(merge, merge, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::findContours(merge, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // filter contours by size deleting the ones smaller than the area defined in the method call, reduces noise in the mask without using morphologicals
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(merge , contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
    }

    //assign variables for access
    final_binary_mask = merge.clone();
    final_mask = getColorMask(merge, parkingSpaceMask);
    final_image = Graphics::maskApplication(src_clean, final_mask);
}
