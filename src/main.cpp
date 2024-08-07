#include <iostream>
#include <filesystem>

#include "../include/BoundingBoxes.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/highgui.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <frames_directory>" << std::endl;
        return -1;
    }

    std::filesystem::path pathSequenceFramesDir = std::filesystem::absolute(argv[1]);
    if (!std::filesystem::exists(pathSequenceFramesDir) || !std::filesystem::is_directory(pathSequenceFramesDir)) {
        std::cerr << "Input directory does not exist or is not a directory." << std::endl;
        return -1;
    }

    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat inputImg = cv::imread(imgPath);
        if (inputImg.empty()) {
            std::cout << "Error opening the image" << std::endl;
            return -1;
        }
        BoundingBoxes BBoxes = BoundingBoxes(inputImg);

        cv::Mat test = BBoxes.getImg();
        cv::imshow("mongus", test);
        cv::waitKey(0);
    }
    return 0;
}