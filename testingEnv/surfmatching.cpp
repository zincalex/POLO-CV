//
// Created by trigger on 8/11/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void matchAndDrawBoundingBox(const Mat& templateImg, const Mat& targetImg, Mat& outputImg, double matchThreshold = 0.75) {
    // Initialize SIFT detector
    Ptr<SIFT> sift = SIFT::create();

    // Detect keypoints and compute descriptors for the template and target images
    vector<KeyPoint> keypointsTemplate, keypointsTarget;
    Mat descriptorsTemplate, descriptorsTarget;
    sift->detectAndCompute(templateImg, noArray(), keypointsTemplate, descriptorsTemplate);
    sift->detectAndCompute(targetImg, noArray(), keypointsTarget, descriptorsTarget);

    // Match descriptors using BFMatcher
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptorsTemplate, descriptorsTarget, matches);

    // Filter matches using the ratio test
    double minDist = 100;
    for (size_t i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
    }

    vector<DMatch> goodMatches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= max(2 * minDist, matchThreshold)) {
            goodMatches.push_back(matches[i]);
        }
    }

    // Extract matched keypoints
    vector<Point2f> templatePoints, targetPoints;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        templatePoints.push_back(keypointsTemplate[goodMatches[i].queryIdx].pt);
        targetPoints.push_back(keypointsTarget[goodMatches[i].trainIdx].pt);
    }

    if (goodMatches.size() >= 4) { // Need at least 4 points to find homography
        // Find homography
        Mat H = findHomography(templatePoints, targetPoints, RANSAC);

        // Get the bounding box from the template image
        vector<Point2f> templateCorners(4);
        templateCorners[0] = Point2f(0, 0);
        templateCorners[1] = Point2f((float)templateImg.cols, 0);
        templateCorners[2] = Point2f((float)templateImg.cols, (float)templateImg.rows);
        templateCorners[3] = Point2f(0, (float)templateImg.rows);
        vector<Point2f> targetCorners(4);

        // Apply homography to get the target corners
        perspectiveTransform(templateCorners, targetCorners, H);

        // Create a bounding rectangle
        Rect boundingBox = boundingRect(targetCorners);
        rectangle(outputImg, boundingBox, Scalar(0, 255, 0), 2);
    }
}

int main() {
    // Load the target image
    Mat targetImg = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence3/frames/2013-03-19_07_25_01.png");
    if (targetImg.empty()) {
        cout << "Could not open or find the target image" << endl;
        return -1;
    }
    Mat outputImg = targetImg.clone();

    // Load multiple template images
    vector<Mat> templates;
    templates.push_back(imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/refs/car1.jpg", IMREAD_GRAYSCALE));
    templates.push_back(imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/refs/car2.jpg", IMREAD_GRAYSCALE));
    templates.push_back(imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/refs/car3.jpg", IMREAD_GRAYSCALE));
    templates.push_back(imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/refs/car4.jpg", IMREAD_GRAYSCALE));

    // Ensure all templates are loaded
    for (size_t i = 0; i < templates.size(); i++) {
        if (templates[i].empty()) {
            cout << "Could not open or find template " << i + 1 << endl;
            return -1;
        }
    }

    // Perform template matching for each template
    for (size_t i = 0; i < templates.size(); i++) {
        matchAndDrawBoundingBox(templates[i], targetImg, outputImg);
    }

    // Display the result
    imshow("Detected Cars", outputImg);
    waitKey(0);

    return 0;
}