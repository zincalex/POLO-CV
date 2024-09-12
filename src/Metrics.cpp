/**
 * @author Alessandro Viespoli 2120824
 */
#include "../include/Metrics.hpp"

Metrics::Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction,
                 const cv::Mat& trueSegmentationMask, const cv::Mat& segmentationColorMask) {
    this->groundTruth = groundTruth;
    this->bBoxesPrediction = bBoxesPrediction;
    this->trueSegmentationMask = trueSegmentationMask;
    this->segmentationColorMask = segmentationColorMask;

    // Building the optinal area mask
    cv::Size imgSize = segmentationColorMask.size();
    cv::Mat mask;
    mask = ImageProcessing::optionalAreaROI(imgSize);
    cv::bitwise_not(mask, this->optionalAreaMask);
}


double Metrics::calculateMeanAveragePrecisionParkingSpaceLocalization() const {
    std::vector<int> totalGroundTruths = {0,0};             // total number of empty and full parking space
    std::vector<std::vector<double>> precisions(2);      // cumulative precision values per class
    std::vector<std::vector<double>> recalls(2);         // cumulative recall values per class

    // AP calculated using the 11 point interpolation method
    std::vector<double> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> averagePrecisions(2, 0);

    // Divide the predictions in the 2 classes -----> 0 no car, 1  car
    std::vector<std::vector<BoundingBox>> sortedPredictionBoxes(2);
    for (const BoundingBox& bBox : bBoxesPrediction)
        bBox.isOccupied() ? sortedPredictionBoxes[1].push_back(bBox) : sortedPredictionBoxes[0].push_back(bBox);

    // Count the actual number of cars
    for (const BoundingBox& trueB : groundTruth)
        if (trueB.getNumber() <= 37) // get only the values of the main two parking spaces (avoiding the upper part)
            trueB.isOccupied() ? totalGroundTruths[1] += 1 : totalGroundTruths[0] += 1;

    // Check the number of classes (in some sequence images, there are no car. Hence, it might happen that no cars are detected)
    std::vector<unsigned int> classes; // vector of indices
    for (unsigned int i = 0; i < 2; ++i) {
        if (totalGroundTruths[i] != 0)
            classes.push_back(i);
    }

    // Calculate the cumulative precisions and recalls for each class
    for (unsigned int& i : classes) {

        unsigned int truePositives = 0;
        unsigned int falsePositives = 0;
        for (const BoundingBox &predictBBox: sortedPredictionBoxes[i]) {
            double bestIOU = -1;
            short bestMatchIndex = -1;

            // In order to determine which parking space is being predicted, we compare each prediction with all the ground truth boxes
            // and keep the greater IoU
            for (unsigned int j = 0; j < groundTruth.size(); ++j) {
                double iou = calculateIoUObjectDetection(predictBBox.getRotatedRect(), groundTruth[j].getRotatedRect());
                if (iou > bestIOU) {
                    bestIOU = iou;
                    bestMatchIndex = static_cast<short>(j);
                }
            }

            // Only if the IoU is above the 0.5 threshold and the prediction is correct, we count is as a true positive
            (bestIOU >= IOU_THRESHOLD && predictBBox.isOccupied() == groundTruth[bestMatchIndex].isOccupied()) ? truePositives++ : falsePositives++;

            // Add the cumulative values
            precisions[i].push_back(static_cast<double>(truePositives) / (truePositives + falsePositives));
            recalls[i].push_back(static_cast<double>(truePositives) / totalGroundTruths[i]);
        }
    }

    // Compute the AP for each class
    for (unsigned int& i : classes) {
        double ap = 0.0;

        for (double recall_level : recall_levels) { // Iterate through each recall level
            double max_precision = 0.0;

            for (unsigned int k = 0; k < recalls[i].size(); ++k)  // for each cumulative value stored
                if (recalls[i][k] >= recall_level)
                    max_precision = std::max(max_precision, precisions[i][k]);

            ap += max_precision;  // Add the maximum precision for this recall level
        }
        averagePrecisions[i] = ap / 11.0;   // 11 Point Interpolation Method
    }

    // Compute mAP
    double mean = 0.0;
    for (unsigned int& i : classes)
        mean += averagePrecisions[i];
    mean /= classes.size();

    return mean;
}


double Metrics::calculateMeanIntersectionOverUnionSegmentation() const {
    unsigned int possibleTotalClasses = 3;
    double totalIoU = 0;
    std::vector<double> iouPerClass;

    // Calculate IoU for each class
    for (unsigned int classId = 0; classId < possibleTotalClasses; ++classId) {
        // Since the GT mask and segmentation mask have different values for identifying a class
        // we pass it to a function in order to make the mask in grayscale and align the convention
        double iou = calculateIoUSegmentation(ImageProcessing::convertColorMaskToGray(trueSegmentationMask),
                                              ImageProcessing::convertColorMaskToGray(segmentationColorMask), classId);

        if (iou != -1) { // Class present
            iouPerClass.push_back(iou);
            totalIoU += iou;
        }
    }

    // Calculate average IoU over all classes
    return totalIoU / static_cast<double>(iouPerClass.size());
}


double Metrics::calculateIoUObjectDetection(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) const {
    std::vector<cv::Point2f> vertices1(4);
    rect1.points(vertices1.data());

    std::vector<cv::Point2f> vertices2(4);
    rect2.points(vertices2.data());

    // Find the intersection of the two rects
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


double Metrics::calculateIoUSegmentation(const cv::Mat& groundTruthMask, const cv::Mat& predictedMask, const unsigned int& classId) const {
    // classId :
    //   - 0 -> background
    //   - 1 -> car inside parking space
    //   - 2 -> car outside parking space
    cv::Mat groundTruthClass = (groundTruthMask == classId); // 255 where there is a match
    cv::Mat predictedClass = (predictedMask == classId);

    // Adjust ground truth for the optional area
    groundTruthClass = groundTruthClass & optionalAreaMask;

    // Calculate intersection and union
    cv::Mat intersectionMask = groundTruthClass & predictedClass;
    cv::Mat unionMask = groundTruthClass | predictedClass;

    // Calculate the number of pixels for each mask
    unsigned long intersectionCount = cv::countNonZero(intersectionMask);
    unsigned long unionCount = cv::countNonZero(unionMask);

    if (unionCount == 0) { // No elements of that class
        return -1;
    }

    // In the case where the G.T has a value while the prediction has not (intersection is zero), and viceversa, the return value is 0
    return static_cast<double>(intersectionCount) / static_cast<double>(unionCount);
}