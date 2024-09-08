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
    std::string framePath = static_cast<std::string>(argv[2]) + "/frames";
    std::string groundTruthPath = static_cast<std::string>(argv[2]) + "/bounding_boxes";
    std::string maskPath = static_cast<std::string>(argv[2]) + "/masks";

    std::filesystem::path pathSequence0FramesDir = std::filesystem::absolute(argv[1]);
    std::filesystem::path pathSequenceFramesDir = std::filesystem::absolute(framePath);
    std::filesystem::path pathGroundTruthDir = std::filesystem::absolute(groundTruthPath);
    std::filesystem::path pathMaskDir = std::filesystem::absolute(maskPath);

    // Check directories
    if (!checkDirectory(pathSequence0FramesDir, "sequence0") || !checkDirectory(pathSequenceFramesDir, "Frames") ||
        !checkDirectory(pathGroundTruthDir, "bounding_boxes") || !checkDirectory(pathMaskDir, "masks")) {
        return -1;
    }


    ParkingSpaceDetector psDetector = ParkingSpaceDetector(pathSequence0FramesDir);
    std::vector<BoundingBox> bBoxes = psDetector.getBBoxes();

    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) { // For each image
        std::string imgPath = iter.path().string();

        std::string filenameWithoutExtension = iter.path().stem().string(); // Removes the ".png"
        std::string xmlDirectory = pathSequenceFramesDir.parent_path().string() + "/bounding_boxes";
        std::string xmlFilename = filenameWithoutExtension + ".xml";
        std::string xmlPath = xmlDirectory + "/" + xmlFilename;


        XMLReader groundTruth = XMLReader(xmlPath);
        cv::Mat parkingImg = cv::imread(imgPath);
        cv::Mat clone = parkingImg.clone();
        ParkingLotStatus parkingStatus = ParkingLotStatus(parkingImg, bBoxes);
        //ParkingLotStatus veritas = ParkingLotStatus(clone, groundTruth.getBBoxes());

        std::cout << "Working on img : " << imgPath << std::endl;
        cv::imshow("Status", parkingStatus.seeParkingLotStatus());
        //cv::imshow("Veritas", veritas.seeParkingLotStatus());
        cv::waitKey(0);
        std::vector<unsigned short> a = parkingStatus.getOccupiedParkingSpaces();

        // First Metric
        cv::Mat zero;
        Metrics metrics = Metrics(groundTruth.getBBoxes(), parkingStatus.getStatusPredictions(), zero);

        //std::cout << "mAP: " << metrics.calculateMeanAveragePrecisionParkingSpaceLocalization() << std::endl;

        // Segmentation

        //Segmentation seg = Segmentation(pathSequence0FramesDir, imgPath);


        // Second Metric

        // 2D Map
        Graphics::applyMap(imgPath, parkingStatus.getOccupiedParkingSpaces());


    }


    return 0;
}