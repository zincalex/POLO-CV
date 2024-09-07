//
// Created by trigger on 9/6/24.
//

#include <opencv2/features2d.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
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
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(bg1_mask, bg1_mask, cv::MORPH_OPEN, kernel);


    return bg1_mask;
}

cv::Mat Segmentation::smallContoursElimination(const cv::Mat& input_mask) {
    cv::Mat out_mask;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(input_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
// Filtro dei contorni per dimensione
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        // Se il contorno è troppo piccolo, lo eliminiamo
        if (area < 100) { // Modifica la soglia a seconda delle dimensioni del tuo dataset
            cv::drawContours(out_mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }
    return out_mask;
}



cv::Mat Segmentation::siftMaskEnhancement(const cv::Mat& starting_mask, const cv::Mat &empty_parking,
                                          const cv::Mat &masked_busy_parking) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_empty, keypoints_with_cars;
    cv::Mat descriptors_empty, descriptors_with_cars;

    sift->detectAndCompute(empty_parking, cv::noArray(), keypoints_empty, descriptors_empty);
    sift->detectAndCompute(masked_busy_parking, cv::noArray(), keypoints_with_cars, descriptors_with_cars);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_empty, descriptors_with_cars, matches);

    double max_dist = 0;
    double min_dist = 100;
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

    // Step 3: Affinamento della maschera preliminare usando i punti chiave non corrispondenti
    cv::Mat mask = starting_mask.clone(); // Usa la maschera dalla sottrazione come base

    // Trova i punti chiave nell'immagine con le auto che non hanno una buona corrispondenza
    for (int i = 0; i < keypoints_with_cars.size(); i++) {
        bool has_match = false;
        for (const auto &match: good_matches) {
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
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15))); // Rimuove i buchi
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30,
                                                                                                    30)));// Ulteriori operazioni morfologiche
    mask = smallContoursElimination(mask);

    return mask;
}

cv::Mat Segmentation::grabCutMask(const cv::Mat& input_mask, const cv::Mat& input_img) {
    cv::Mat grabcut_mask = input_mask.clone();
    cv::Mat start = input_img.clone();

    grabcut_mask.setTo(cv::GC_BGD, grabcut_mask == 0);
    grabcut_mask.setTo(cv::GC_PR_FGD, grabcut_mask == 255);

    cv::Mat bgdModel, fgdModel;

    try {
        cv::grabCut(start, grabcut_mask, cv::Rect(), bgdModel, fgdModel, 30, cv::GC_INIT_WITH_MASK);
        start.copyTo(start, grabcut_mask);
        cv::grabCut(start, grabcut_mask, cv::Rect(), bgdModel, fgdModel, 30, cv::GC_INIT_WITH_MASK);
    }
    catch (const cv::Exception &e) {
        std::cout << "no foreground detected" << std::endl;
    }
    cv::compare(grabcut_mask, cv::GC_PR_FGD, grabcut_mask, cv::CMP_EQ);
    // Filtro dei contorni per dimensione
    cv::morphologyEx(grabcut_mask, grabcut_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    cv::morphologyEx(grabcut_mask, grabcut_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

    grabcut_mask = smallContoursElimination(grabcut_mask);

    cv::Mat fgMask = cv::Mat::zeros(start.size(), start.type());
    for (int x = 0; x < start.rows; x++) {  // Iterate over rows (height)
        for (int y = 0; y < start.cols; y++) {  // Iterate over columns (width)
            if ((int) grabcut_mask.at<uchar>(cv::Point(y, x)) == 255) {
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[0] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[1] = 255;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[2] = 0;
            } else {
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[0] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[1] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[2] = 0;
            }
        }
    }

    return  grabcut_mask;
}



Segmentation::Segmentation(const std::filesystem::path &emptyFramesDir, const std::filesystem::path &FramesDir) {
    for (const auto &iter: std::filesystem::directory_iterator(FramesDir)){
        std::string imgPathBusy = iter.path().string();
        cv::Mat parking_with_cars_col = cv::imread(imgPathBusy);
        cv::Mat parking_with_cars;
        cv::cvtColor(parking_with_cars_col, parking_with_cars, cv::COLOR_BGR2GRAY);
        cv::Mat src_clean = parking_with_cars_col.clone();//used only for final mask application

        cv::Mat mask;
        cv::Mat empty_parking = averageEmptyImages(emptyFramesDir);

        mask = backgroundSubtractionMask(empty_parking, parking_with_cars);

        parking_with_cars.copyTo(parking_with_cars, mask);

        mask = siftMaskEnhancement(mask, empty_parking, parking_with_cars);

        parking_with_cars_col.copyTo(parking_with_cars_col, mask);

        //parking_with_cars_col = ImageProcessing::applyCLAHE(parking_with_cars_col);
        ImageProcessing::adjustContrast(parking_with_cars_col, 2, -50);

        mask = grabCutMask(mask, parking_with_cars_col);

        addWeighted(mask, 1, src_clean, 0.5, 0, src_clean);

        cv::imshow("segmentation", src_clean);

        cv::waitKey(0);
    }
}


