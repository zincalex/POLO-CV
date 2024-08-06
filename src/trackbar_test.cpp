#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

typedef struct Params{
    Mat img, gray;
    Mat dest, roi;


    int x_coord_center = 0;
    const int x_coord_center_max = 1280;
    int y_coord_center = 0;
    const int y_coord_center_max = 720;
    int height = 0;
    const int maxheight = 720;
    int width = 0;
    const int maxwidth = 1280;
    int angolo = -90;
    const int max_angolo = 90;
} gParams;


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
        cv::namedWindow("mongus", cv::WINDOW_AUTOSIZE);
        cv::imshow("mongus", test);
        cv::waitKey(0);
    }

    // Apply Bilateral Filter to reduce noise while keeping edges sharp



    CannyParams userdata;

    userdata.img = imread(image_name);

    if(userdata.img.empty()){
        std::cout << "Error reading input file, check working directory";
        return(1);
    }

    userdata.dest.create(userdata.img.rows, userdata.img.cols, userdata.img.type());


    namedWindow("Output with ROIs", WINDOW_AUTOSIZE);


    createTrackbar( "X_Center", "Output with ROIs", &userdata.x_coord_center, userdata.x_coord_center_max, RoiFunct, &userdata);
    createTrackbar( "Y_Center", "Output with ROIs", &userdata.y_coord_center, userdata.y_coord_center_max, RoiFunct, &userdata);
    createTrackbar( "Height", "Output with ROIs", &userdata.height, userdata.maxheight, RoiFunct, &userdata);
    createTrackbar( "Width", "Output with ROIs", &userdata.width, userdata.maxwidth, RoiFunct, &userdata);
    createTrackbar( "Angolo", "Output with ROIs", &userdata.angolo, userdata.max_angolo, RoiFunct, &userdata);

    RoiFunct(0, &userdata);

    waitKey(0);

    // Find contours


    return 0;
}

