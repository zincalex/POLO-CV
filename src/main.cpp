
#include <iostream>
#include <filesystem>

#include "../include/BoundingBox.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/highgui.hpp"

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <frames_directory>" << std::endl;
        return 1;
    }

    std::filesystem::path pathSequenceFramesDir = std::filesystem::absolute(argv[1]);
    if (!std::filesystem::exists(pathSequenceFramesDir) || !std::filesystem::is_directory(pathSequenceFramesDir)) {
        std::cerr << "Input directory does not exist or is not a directory." << std::endl;
        return 1;
    }

    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) {
        std::string imgPath = iter.path().string();
        cv::Mat img = cv::imread(imgPath);
        BoundingBox hello = BoundingBox(img);

        cv::Mat test = hello.getImg();
        cv::namedWindow("mongus", cv::WINDOW_AUTOSIZE);
        cv::imshow("mongus", test);
        cv::waitKey(0);
    }

}