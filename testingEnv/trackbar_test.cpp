#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "../include/BoundingBoxes.hpp"
using namespace cv;
using namespace std;

//paste the method you want to test
Mat niBlack_thresholding(const cv::Mat& input, const int& blockSize, const double& k) {
    cv::Mat gray_image, niblack;
    cv::cvtColor(input, gray_image, cv::COLOR_BGR2GRAY);
    cv::ximgproc::niBlackThreshold(gray_image, niblack, 255, cv::THRESH_BINARY, blockSize, k, cv::ximgproc::BINARIZATION_NIBLACK);
    return niblack;
}

//struct with user defined parameters
typedef struct Params{

    string imagePath;
    //define Mat needed to run the method to test
    Mat img;
    Mat dest;

    //define variable min and max to be changed with trackbars
    int blockSize = 3;
    const int blockSizeMax = 1000;
    int k = 0;
    const int maxK = 50;
} gParams;

//preprocessing needed to create the ROIs making the environment similar to the main program
Mat createROI(const cv::Mat& input) { // We focus the analysis of the image on the parking lots
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    // Define ROIs
    std::vector<cv::RotatedRect> rois;
    rois.push_back(cv::RotatedRect(cv::Point2f(572, 317), cv::Size2f(771, 282), 58));
    rois.push_back(cv::RotatedRect(cv::Point2f(950, 200), cv::Size2f(165, 710), -54));
    rois.push_back(cv::RotatedRect(cv::Point2f(1136, 105), cv::Size2f(73, 467), 118));

    std::vector<cv::RotatedRect> black_rois;        // More ad hoc ROI in order to refine the ROI selected
    black_rois.push_back(cv::RotatedRect(cv::Point2f(799, 343), cv::Size2f(1227, 125), 46));
    black_rois.push_back(cv::RotatedRect(cv::Point2f(326, 3), cv::Size2f(62, 113), 50));
    black_rois.push_back(cv::RotatedRect(cv::Point2f(861, 25), cv::Size2f(552, 64), 33));

    for (const auto& roiRect : rois) {
        cv::Point2f vertices[4];
        std::vector<cv::Point> contour;

        roiRect.points(vertices);    // Store the vertices of the ROI
        for (auto vertex : vertices) { contour.push_back(vertex); }
        cv::fillConvexPoly(mask, contour, cv::Scalar(255));
    }

    for (const auto& blackRoiRect : black_rois) {
        cv::Point2f vertices[4];
        std::vector<cv::Point> contour;

        blackRoiRect.points(vertices);
        for (auto vertex : vertices) { contour.push_back(vertex); }
        cv::fillConvexPoly(mask, contour, cv::Scalar(0));
    }

    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++)
            if (mask.at<uchar>(y, x) == 255)
                result.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(y, x);

    return result;
}

static void CallbackFunct(int, void* userdata){

    //get parameters from user defined struct
    Params &params = *((Params*)userdata);

    //handles the odd-only parameters always adding 1 if the trackbar is set to an even value, comment if not needed
    if (params.blockSize % 2 == 0) {
        params.blockSize++;
    }

    //call the function you want to test
    params.dest = niBlack_thresholding(params.img, params.blockSize, static_cast<double>(params.k));


    // Display the output image
    imshow(params.imagePath, params.dest);

}


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
    gParams userdata;



    for (const auto& iter : std::filesystem::directory_iterator(pathSequenceFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat inputImg = cv::imread(imgPath);
        if (inputImg.empty()) {
            std::cout << "Error opening the image" << std::endl;
            return -1;
        }

        //load the image to process in the struct to pass to callback function
        Mat roiInput = createROI(inputImg);
        userdata.imagePath = imgPath;
        userdata.img = roiInput;
        userdata.dest.create(userdata.img.rows, userdata.img.cols, userdata.img.type());
        namedWindow(imgPath, WINDOW_AUTOSIZE);

        //create all the trackbars with the relative parameters
        createTrackbar( "Blocksize", imgPath, &userdata.blockSize, userdata.blockSizeMax, CallbackFunct, &userdata);
        createTrackbar( "K", imgPath, &userdata.k, userdata.maxK, CallbackFunct, &userdata);

        CallbackFunct(0, &userdata);

        waitKey(0);
    }



    return 0;
}

