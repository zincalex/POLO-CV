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
    }
    cv::Mat empty_parking = cv::Mat::zeros(empty_images[0].size(), CV_32FC1);
    for (const auto &img: empty_images) {
        cv::Mat temp;
        img.convertTo(temp, CV_32FC1);
        empty_parking += temp;
    }
    empty_parking /= empty_images.size();

    empty_parking.convertTo(empty_parking, CV_8UC1);
    return empty_parking;
}

cv::Mat Segmentation::backgroundSubtractionMask(const cv::Mat &empty_parking, const cv::Mat &busy_parking) {
    cv::Mat diff;
    cv::absdiff(busy_parking, empty_parking, diff);
    cv::Mat bg1_mask;
    cv::threshold(diff, bg1_mask, 50, 255, cv::THRESH_BINARY);

    // Affina la maschera usando operazioni morfologiche
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

// Filtro dei contorni per dimensione
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        // Se il contorno è troppo piccolo, lo eliminiamo
        if (area < minArea) { // Modifica la soglia a seconda delle dimensioni del tuo dataset
            cv::drawContours(in_mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }
    return in_mask;
}

cv::Ptr<cv::BackgroundSubtractorMOG2> Segmentation::trainBackgroundModel(const std::vector<cv::String>& backgroundImages) {
    // Crea il sottrattore di background MOG2
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2();

    for (const auto &imagePath: backgroundImages) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Errore nel caricamento dell'immagine: " << imagePath << std::endl;
            continue;
        }

        // Applica il modello di background all'immagine di background per "allenarlo"
        cv::Mat fgMask;
        mog2->apply(image, fgMask, 0.55);
    }
    return mog2;
}

cv::Mat Segmentation::getForegroundMaskMOG2(cv::Ptr<cv::BackgroundSubtractorMOG2> &mog2, cv::Mat &busy_parking) {
    cv::Mat image_src = busy_parking.clone();
    cv::Mat fgMask;
    mog2->apply(image_src, fgMask, 0);  // Usa un learning rate di 0 per non aggiornare piÃ¹ il background
    fgMask = smallContoursElimination(fgMask,100);

    cv::waitKey(0);

    for (int x = 0; x < fgMask.rows; x++) {  // Iterate over rows (height)
        for (int y = 0; y < fgMask.cols; y++) {  // Iterate over columns (width)
            if ((unsigned char) fgMask.at<uchar>(cv::Point(y, x)) == 127) {
                fgMask.at<uchar>(cv::Point(y, x)) = 0;
            }
        }
    }

    fgMask = smallContoursElimination(fgMask,100);

    //cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// Crea un'immagine su cui disegnare
    cv::Mat drawing = cv::Mat::zeros(fgMask.size(), CV_8UC1);

// Per ogni contorno trovato, riempi il poligono se è convesso
    for (size_t i = 0; i < contours.size(); i++) {
        cv::fillPoly(drawing, contours[i], cv::Scalar(255, 255, 255)); // Riempi con colore verde
    }

    return drawing;
}

cv::Mat Segmentation::getBBoxMask(const std::vector<BoundingBox> &parkingBBoxes, cv::Mat& target) {

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

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(car_fgMask, labels,stats, centroids);

    cv::Mat out_mask = cv::Mat::zeros(car_fgMask.size(), CV_8UC3);

    for(int label = 1; label < num_labels; label++){
        cv::Mat current_object = (labels == label);

        cv::Mat intersect;
        cv::bitwise_and(current_object, parking_mask, intersect);

        if(cv::countNonZero(intersect)>0){
            out_mask.setTo(PARKED_CAR, current_object);
        } else{
            out_mask.setTo(ROAMING_CAR, current_object);
        }
    }
    return out_mask;
}

cv::Mat Segmentation::seeSegmentationResult() {
    return final_image;
}

cv::Mat Segmentation::getSegmentationMask() {
    return final_mask;
}

Segmentation::Segmentation(const std::filesystem::path &emptyFramesDir, const std::filesystem::path &mogTrainingDir,const std::vector<BoundingBox>& parkingBBoxes,const std::string& imageName) {
        cv::Mat parking_with_cars_col = cv::imread(imageName);
        cv::Mat parking_with_cars;
        cv::cvtColor(parking_with_cars_col, parking_with_cars, cv::COLOR_BGR2GRAY);
        cv::Mat src_clean = parking_with_cars_col.clone();//used only for final mask application
        cv::Mat mask;
        cv::Mat empty_parking = averageEmptyImages(emptyFramesDir);
        cv::Mat parking_masked;
        std::vector<cv::String> backgroundImages;
        cv::glob(mogTrainingDir, backgroundImages);
        cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = trainBackgroundModel(backgroundImages);
        cv::Mat bgSubctMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
        cv::Mat parkingSpaceMask = cv::Mat::zeros(parking_with_cars.size(), CV_8UC1);
        parkingSpaceMask = getBBoxMask(parkingBBoxes, parkingSpaceMask);


        cv::Mat Mog2Mask = getForegroundMaskMOG2(mog2, parking_with_cars_col);
        mask = backgroundSubtractionMask(empty_parking, parking_with_cars);
        cv::bitwise_and(mask, Mog2Mask, bgSubctMask);



        cv::morphologyEx(bgSubctMask, bgSubctMask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 16)));
        cv::morphologyEx(bgSubctMask, bgSubctMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        bgSubctMask = smallContoursElimination(bgSubctMask, 1500);

        final_mask = getColorMask(bgSubctMask, parkingSpaceMask);
        final_image = Graphics::maskApplication(src_clean, final_mask);

}






