#include "../include/Metrics.hpp"

bool isWithinRadius(const cv::Point& predictedCenter, const cv::Point& trueCenter, const double& radius) {
    double distance = std::sqrt(std::pow(predictedCenter.x - trueCenter.x, 2) + std::pow(predictedCenter.y - trueCenter.y, 2));
    return distance <= radius;
}

double calculateIoU(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> vertices1(4);
    rect1.points(vertices1);

    std::vector<cv::Point2f> vertices2(4);
    rect2.points(vertices2);

    // Find the intersection area
    std::vector<cv::Point2f> intersectionPoints;
    cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);

    // Calculate the intersection area using contour area
    double intersectionArea = cv::contourArea(intersectionPoints);

    double area1 = rect1.size.area();
    double area2 = rect2.size.area();

    return intersectionArea / (area1 + area2 - intersectionArea);
}

double Metrics::calculateMeanAveragePrecisionParkingSpaceLocalization() const {
    const double RADIUS = 45.0;
    const double IOU_THRESHOLD = 0.5;

    std::vector<std::vector<double>> recalls(2);
    std::vector<std::vector<double>> precisions(2);
    std::vector<double> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::vector<double> average_precision(2, 0);

    // Divide the classification in the 2 classes -----> 0 no car, 1  car
    std::vector<std::vector<BoundingBox>> sortedPredictionBoxes(2);
    std::vector<std::vector<BoundingBox>> sortedGroundTruth(2);
    for (const BoundingBox& bBox : bBoxesPrediction)
        bBox.isOccupied() ? sortedPredictionBoxes[1].push_back(bBox) : sortedPredictionBoxes[0].push_back(bBox);
    for (const BoundingBox& trueBox : groundTruth)
        trueBox.isOccupied() ? sortedGroundTruth[1].push_back(trueBox): sortedGroundTruth[0].push_back(trueBox);



    // Calculate the cumulative precisions and recalls for each class
    for (unsigned int i = 0; i < 2; ++i) {
        unsigned int totalGroundTruths = sortedGroundTruth[i].size();
        unsigned int truePositives = 0;
        unsigned int falsePositives = 0;

        for (const BoundingBox &predictBBox: sortedPredictionBoxes[i]) {
            for (const BoundingBox &trueBBox: sortedGroundTruth[i]) {

                // Same parking spot
                if (isWithinRadius(predictBBox.getCenter(), trueBBox.getCenter(), RADIUS)) {
                    std::cout << "Pred box  " << predictBBox.getNumber() << " matched with ground truth " << trueBBox.getNumber() << std::endl;
                    double iou = calculateIoU(predictBBox.getRotatedRect(), trueBBox.getRotatedRect());
                    std::cout << "IOU: " << iou << std::endl;
                    (iou >= IOU_THRESHOLD) ? truePositives++ : falsePositives++;
                    precisions[i].push_back(truePositives / (truePositives + falsePositives));
                    recalls[i].push_back(truePositives / totalGroundTruths);
                    break;
                }

            }
        }
    }


    // For each class
    for (int i = 0; i < 1; i++) {
        double ap = 0.0;

        // Iterate through each recall level
        for (int j = 0; j < recall_levels.size(); j++) {
            double recall_level = recall_levels[j];
            // Find the maximum precision for recall >= recall_level
            double max_precision = 0.0;
            for (int k = 0; k < recalls.size(); k++) {
                if (recalls[i][k] >= recall_level) {
                    max_precision = std::max(max_precision, precisions[i][k]);
                }
            }
            ap += max_precision;  // Store the maximum precision for this recall level
        }
        average_precision[i] = ap / 11.0;   // 11 Point Interpolation Method

        std::cout << "avgP: " << average_precision[i] << std::endl;
    }


    // Compute mAP
    double mean = 0.0;   // Initialize mean average precision
    for (int i = 0; i < 2; i++)
        mean += average_precision[i];
    mean /= 2;

    return mean;
}



Metrics::Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction, const cv::Mat& segmentationColorMask) {
    this->groundTruth = groundTruth;
    this->bBoxesPrediction = bBoxesPrediction;
    this->segmentationColorMask = segmentationColorMask;
    totBoundingBoxes = bBoxesPrediction.size();
}