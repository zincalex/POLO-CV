/**
 * @author Alessandro Viespoli 2120824
 */

#include <iostream>
#include <filesystem>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../include/ParkingSpaceDetector.hpp"
#include "../include/ParkingLotStatus.hpp"
#include "../include/XMLReader.hpp"
#include "../include/Segmentation.hpp"
#include "../include/Metrics.hpp"
#include "../include/Graphics.hpp"

bool checkDirectory(const std::filesystem::path& dirPath, const std::string& dirName) {
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << dirName << " directory does not exist or is not a directory." << std::endl;
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: <sequence0_frames_directory> <sequence_directory>" << std::endl;
        return -1;
    }


    // Directories paths
    // Parking space localization paths
    std::filesystem::path pathSequence0FramesDir = std::filesystem::absolute(argv[1]);

    // Images paths
    std::string framePath = static_cast<std::string>(argv[2]) + "/frames";
    std::filesystem::path pathSequenceFramesDir = std::filesystem::absolute(framePath);

    // Ground truth paths
    std::string groundTruthPath = static_cast<std::string>(argv[2]) + "/bounding_boxes";
    std::string maskPath = static_cast<std::string>(argv[2]) + "/masks";
    std::filesystem::path pathGroundTruthDir = std::filesystem::absolute(groundTruthPath);
    std::filesystem::path pathMaskDir = std::filesystem::absolute(maskPath);

    // Segmentation paths
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::string trainingPath = static_cast<std::string>(currentPath) + "/ParkingLot_dataset/mog2_training_sequence";
    std::filesystem::path trainingDir = std::filesystem::absolute(trainingPath);

    // Check directories
    if (!checkDirectory(pathSequence0FramesDir, "sequence0") || !checkDirectory(pathSequenceFramesDir, "Frames") ||
        !checkDirectory(pathGroundTruthDir, "bounding_boxes") || !checkDirectory(pathMaskDir, "masks") ||
        !checkDirectory(trainingDir, "mog2_training_sequence")) {
        return -1;
    }


    // PARKING SPACE LOCALIZATION
    ParkingSpaceDetector psDetector = ParkingSpaceDetector(pathSequence0FramesDir);
    std::vector<BoundingBox> bBoxes = psDetector.getBBoxes();


    // CLASSIFICATION & SEGMENTATION
    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) { // For each image

        // More paths
        std::string imgPath = iter.path().string();
        std::string filenameWithoutExtension = iter.path().stem().string(); // Removes the ".png"
        std::string filenameExtension = iter.path().filename().string();
        std::string xmlFilename = filenameWithoutExtension + ".xml";
        std::string xmlPath = pathGroundTruthDir.string() + "/" + xmlFilename;
        std::string groundTruthMaskPath = pathMaskDir.string() + "/" + filenameExtension;

        // Exctract the parking space state from the ground truth
        XMLReader groundTruth = XMLReader(xmlPath);
        cv::Mat segmentationGTMask = cv::imread(groundTruthMaskPath); //BGR image

        // Segmentation
        Segmentation seg = Segmentation(pathSequence0FramesDir, trainingDir ,bBoxes,imgPath);

        // Car detection
        cv::Mat parkingImg = cv::imread(imgPath);
        ParkingLotStatus parkingStatus = ParkingLotStatus(parkingImg, bBoxes, seg.getMOG2Labmask());

        // Metrics
        Metrics metrics = Metrics(groundTruth.getBBoxes(), parkingStatus.getStatusPredictions(), segmentationGTMask, seg.getSegmentationMaskWithClasses());

        // 2D Map 
        cv::Mat clone = parkingImg.clone();
        Graphics::applyMap(clone, parkingStatus.getOccupiedParkingSpaces());

        // Show results
        std::cout << "\n Working on image " << filenameExtension  << " for the " << framePath << std::endl;
        cv::imshow("Predicted parking lot status", parkingStatus.seeParkingLotStatus());
        cv::waitKey(0);
        cv::imshow("Segmentation", seg.getSegmentationResult());
        cv::waitKey(0);
        std::cout << "----------------     METRICS     ----------------" << std::endl;
        std::cout << "Parking space localization  ----> mAP  = " << metrics.calculateMeanAveragePrecisionParkingSpaceLocalization() << std::endl;
        std::cout << "Car segmentation            ----> mIoU = " << metrics.calculateMeanIntersectionOverUnionSegmentation() << std::endl;
        cv::waitKey(0);
        cv::imshow("2DMap", clone);
        cv::waitKey(0);
    }
    return 0;
}