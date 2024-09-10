#include "../include/Metrics.hpp"

double calculateIoUObjectDetection(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    std::vector<cv::Point2f> vertices1(4);
    rect1.points(vertices1.data());

    std::vector<cv::Point2f> vertices2(4);
    rect2.points(vertices2.data());

    // Find the intersection area
    std::vector<cv::Point2f> intersectionPoints;
    int intersectionResult = cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);

    if (intersectionResult == cv::INTERSECT_NONE) // cv::countourArea gives problem if feed with no points that intersect
        return 0.0;

    // Calculate the intersection area using contour area
    double intersectionArea = cv::contourArea(intersectionPoints);

    double area1 = rect1.size.area();
    double area2 = rect2.size.area();

    return intersectionArea / (area1 + area2 - intersectionArea);
}


double Metrics::calculateMeanAveragePrecisionParkingSpaceLocalization() const {

    const bool DEBUG = false;
    const double IOU_THRESHOLD = 0.5;

    std::vector<int> totalGroundTruths = {0,0};
    std::vector<std::vector<double>> recalls(2);
    std::vector<std::vector<double>> precisions(2);
    std::vector<double> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::vector<double> average_precision(2, 0);

    // Divide the classification in the 2 classes -----> 0 no car, 1  car
    std::vector<std::vector<BoundingBox>> sortedPredictionBoxes(2);
    std::vector<std::vector<BoundingBox>> sortedGroundTruth(2);
    for (const BoundingBox& bBox : bBoxesPrediction)
        bBox.isOccupied() ? sortedPredictionBoxes[1].push_back(bBox) : sortedPredictionBoxes[0].push_back(bBox);
    for (const BoundingBox& trueB : groundTruth)
        if (trueB.getNumber() <= 37)
            trueB.isOccupied() ? totalGroundTruths[1] += 1 : totalGroundTruths[0] += 1;

    // Check the number of classes (in some sequence images, there are no car. Hence, it might happen that no cars are dected)
    std::vector<unsigned int> classes;
    for (unsigned int i = 0; i < 2; ++i) {
        if (totalGroundTruths[i] != 0)
            classes.push_back(i);
    }



    // Calculate the cumulative precisions and recalls for each class
    for (unsigned int& i : classes) {
        unsigned int truePositives = 0;
        unsigned int falsePositives = 0;

        if (DEBUG) std::cout << "Working for cars that are : " << i << std::endl;
        if (DEBUG) std::cout << "Number of true values here : " << totalGroundTruths[i] << std::endl;

        for (const BoundingBox &predictBBox: sortedPredictionBoxes[i]) {
            short bestMatch = -1;
            short bestMatchIndex = -1;
            double bestIOU = -1;
            for (unsigned int j = 0; j < groundTruth.size(); ++j) {
                double iou = calculateIoUObjectDetection(predictBBox.getRotatedRect(), groundTruth[j].getRotatedRect());
                if (iou > bestIOU) {
                    bestIOU = iou;
                    bestMatchIndex = static_cast<short>(j);
                }
            }
            if (DEBUG) std::cout << "Car " << predictBBox.getNumber() << " matched with position " << groundTruth[bestMatchIndex].getNumber()  << " with iou " << bestIOU << std::endl;
            (bestIOU >= IOU_THRESHOLD && predictBBox.isOccupied() == groundTruth[bestMatchIndex].isOccupied()) ? truePositives++ : falsePositives++;
            precisions[i].push_back(static_cast<double>(truePositives) / (truePositives + falsePositives));
            recalls[i].push_back(static_cast<double>(truePositives) / totalGroundTruths[i]);
        }

        if (DEBUG) std::cout << "Class " << i << " ---->   " << "TP : " << truePositives << ", FP : " << falsePositives << std::endl;
        if (DEBUG) std::cout << "------------------ " << std::endl;
    }

    if (DEBUG) {
        for (unsigned int i = 0; i < precisions.size(); ++i) {
            std::cout << "Class " << i << " Precisions: ";
            for (const auto &p: precisions[i]) {
                std::cout << p << " ";
            }
            std::cout << std::endl;
        }
        for (unsigned int i = 0; i < recalls.size(); ++i) {
            std::cout << "Class " << i << " Recalls: ";
            for (const auto& r : recalls[i]) {
                std::cout << r << " ";
            }
            std::cout << std::endl;
        }
    }


    // For each class
    for (unsigned int& i : classes) {
        double ap = 0.0;

        // Iterate through each recall level
        for (double recall_level : recall_levels) {
            double max_precision = 0.0;

            for (unsigned int k = 0; k < recalls[i].size(); ++k)
                if (recalls[i][k] >= recall_level)
                    max_precision = std::max(max_precision, precisions[i][k]);

            if (DEBUG) std::cout << "Max precision " << max_precision << " for recall level " << recall_level << std::endl;

            ap += max_precision;  // Add the maximum precision for this recall level
        }
        average_precision[i] = ap / 11.0;   // 11 Point Interpolation Method

        if (DEBUG) std::cout << "avgP: " << average_precision[i] << std::endl;
    }


    // Compute mAP
    double mean = 0.0;   // Initialize mean average precision
    for (unsigned int& i : classes)
        mean += average_precision[i];
    mean /= classes.size();

    return mean;
}



double calculateIoUSegmentation(const cv::Mat& groundTruthMask, const cv::Mat& predictedMask, const unsigned int& classId) {
    cv::Mat groundTruthClass = (groundTruthMask == classId); // 255 where there is a match
    cv::Mat predictedClass = (predictedMask == classId);

    // Calculate intersection and union
    cv::Mat intersectionMask = (groundTruthClass & predictedClass);
    cv::Mat unionMask = (groundTruthClass | predictedClass);

    unsigned long intersectionCount = cv::countNonZero(intersectionMask);
    unsigned long unionCount = cv::countNonZero(unionMask);

    if (unionCount == 0) { // No elements of that class
        return -1;
    }

    // In the case where the G.T has a value while the prediction has not (intersection is zero), and viceversa, the return value is 0
    return static_cast<double>(intersectionCount) / static_cast<double>(unionCount);
}


double Metrics::calculateMeanIntersectionOverUnionSegmentation() const {

    int possibleTotalClasses = 3;
    double totalIoU = 0;
    std::vector<double> iouPerClass;

    // Calculate IoU for each class
    for (unsigned int classId = 0; classId < possibleTotalClasses; ++classId) {
        double iou = calculateIoUSegmentation(ImageProcessing::convertColorMaskToGray(trueSegmentationMask),
                                              ImageProcessing::convertColorMaskToGray(segmentationColorMask), classId);

        if (iou != -1) {
            iouPerClass.push_back(iou);
            totalIoU += iou;
        }
    }

    // Calculate average IoU over all classes
    return totalIoU / static_cast<double>(iouPerClass.size());
}



Metrics::Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction,
                 const cv::Mat& trueSegmentationMask, const cv::Mat& segmentationColorMask) {
    this->groundTruth = groundTruth;
    this->bBoxesPrediction = bBoxesPrediction;
    this->trueSegmentationMask = trueSegmentationMask;
    this->segmentationColorMask = segmentationColorMask;
}