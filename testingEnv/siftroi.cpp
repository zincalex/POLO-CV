#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <filesystem>
#include "opencv2/ximgproc.hpp"

cv::Mat mergeMask(const std::vector<cv::Mat>& masks){
    if(masks.empty()){
        throw std::runtime_error("No mask provided, check parameters");
    }


    cv::Mat merged = masks[0].clone();

    for (size_t i = 1; i < masks.size(); ++i) {
        bitwise_or(merged, masks[i], merged);
    }

    return merged;
}

int main() {

    cv::Mat parking_with_cars_col = cv::imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence3/frames/2013-03-19_07_05_01.png");
    cv::Mat parking_with_cars;
    cv::cvtColor(parking_with_cars_col, parking_with_cars, cv::COLOR_BGR2GRAY);
    if (parking_with_cars.empty()) {
        std::cerr << "Errore nel caricamento delle immagini" << std::endl;
        return -1;
    }

    std::string emptyFramesDir ="/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence0/frames";
    cv::Mat empty_parking = cv::Mat::zeros(parking_with_cars.size(), CV_32FC1);
    std::vector<cv::Mat> empty_images;
    for (const auto& iter : std::filesystem::directory_iterator(emptyFramesDir)) {
        std::string imgPath = iter.path().string();
        cv::Mat input = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cerr << "Error opening the image" << std::endl;
        }
        empty_images.push_back(input);
    }

    for (const auto& img : empty_images) {
        cv::Mat temp;
        img.convertTo(temp, CV_32FC1);
        empty_parking += temp;
    }
    empty_parking /= empty_images.size();

    empty_parking.convertTo(empty_parking, CV_8UC1);

    // Step 1: Sottrazione del background
    cv::Mat diff;
    cv::absdiff(parking_with_cars, empty_parking, diff);

    // Applica una sogliatura sulla differenza
    cv::Mat thresh;
    cv::threshold(diff,thresh, 50, 255, cv::THRESH_BINARY);

    // Affina la maschera usando operazioni morfologiche
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

    // Visualizza la maschera preliminare
    cv::imshow("Background Subtraction Mask", thresh);

    parking_with_cars.copyTo(parking_with_cars, thresh);

    // Step 2: Segmentazione con SIFT (affinamento)
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_empty, keypoints_with_cars;
    cv::Mat descriptors_empty, descriptors_with_cars;

    sift->detectAndCompute(empty_parking, cv::noArray(), keypoints_empty, descriptors_empty);
    sift->detectAndCompute(parking_with_cars, cv::noArray(), keypoints_with_cars, descriptors_with_cars);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_empty, descriptors_with_cars, matches);

    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptors_empty.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_empty.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    cv::Mat img_matches;
    cv::drawMatches(empty_parking, keypoints_empty, parking_with_cars, keypoints_with_cars, good_matches, img_matches);
    cv::imshow("Good Matches", img_matches);

    // Step 3: Affinamento della maschera preliminare usando i punti chiave non corrispondenti
    cv::Mat mask = thresh.clone(); // Usa la maschera dalla sottrazione come base

    // Trova i punti chiave nell'immagine con le auto che non hanno una buona corrispondenza
    for (int i = 0; i < keypoints_with_cars.size(); i++) {
        bool has_match = false;
        for (const auto& match : good_matches) {
            if (match.trainIdx == i) {
                has_match = true;
                break;
            }
        }
        if (!has_match) {
            // Affina la maschera aggiungendo i punti chiave che non corrispondono
            cv::circle(mask, keypoints_with_cars[i].pt, 5, cv::Scalar(255), cv::FILLED);
        }
    }

    // Affinamento finale della maschera
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;


    // Ulteriori operazioni morfologiche
    //cv::Mat kernelfinal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15))); // Rimuove i buchi
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30)));// Ulteriori operazioni morfologiche
    //cv::morphologyEx(mask, mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10)));


// Trova i contorni nella maschera
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
// Filtro dei contorni per dimensione
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        // Se il contorno è troppo piccolo, lo eliminiamo
        if (area < 100) { // Modifica la soglia a seconda delle dimensioni del tuo dataset
            cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }

    cv::imshow("Mask after refinement", mask);


    // Applica la maschera affinata all'immagine con le auto
    cv::Mat image;
    parking_with_cars_col.copyTo(image, mask);


    mask.setTo(cv::GC_BGD, mask == 0);
    mask.setTo(cv::GC_PR_FGD, mask == 255);

    cv::Mat bgdModel, fgdModel;
    try{cv::grabCut(parking_with_cars_col, mask, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);}
    catch(const cv::Exception& e) {
        std::cout << "no foreground detected" << std::endl;
    }
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
    cv::imshow("Mask after grabcut", mask);
    cv::imshow("Segmented Cars", image);

    cv::waitKey(0);
    return 0;
}
