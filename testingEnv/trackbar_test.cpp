#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "../include/ParkingSpaceDetector.hpp"
using namespace cv;
using namespace std;


//paste the method you want to test
Mat niBlack_thresholding(const cv::Mat& input,int& thresh, int& minL, int& maxLGap ) {
    cv::Mat sugoi, roiGray, roiCanny;

    cvtColor(input, roiGray, cv::COLOR_BGR2GRAY);
    GaussianBlur(roiGray, roiGray, cv::Size(5, 5), 0);
    int blockSize = 5; // Size of the pixel neighborhood used to calculate the threshold
    int C = 2;          // Constant subtracted from the mean or weighted mean
    cv::adaptiveThreshold(roiGray, roiGray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);
    cv::bitwise_not(roiGray, roiGray);
    cv::medianBlur(roiGray, sugoi, 3);
    cvtColor(input, roiGray, cv::COLOR_BGR2GRAY);
    GaussianBlur(roiGray, roiGray, cv::Size(5, 5), 0);
    Canny(roiGray, roiCanny, 100, 100 * 22, 5);
    cv::Mat mask = sugoi | roiCanny;


    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(mask, hough_lines, 1, CV_PI / 180, thresh, minL, maxLGap);
    cv::Mat hough_lines_image = input.clone();
    for (auto l: hough_lines) {
        cv::line(hough_lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255));
    }
    return hough_lines_image;
}

//struct with user defined parameters
typedef struct Params{

    string imagePath;
    //define Mat needed to run the method to test
    Mat img;
    Mat dest;

    //define variable min and max to be changed with trackbars
    int thresh = 0;
    const int threshMax = 80;

    int minL = 0;
    const int minLMax = 100;
    int maxLGap = 0;
    const int maxLGapMax = 60;
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
    if (params.thresh % 2 == 0) {
        params.thresh++;
    }
    if (params.minL % 2 == 0) {
        params.minL++;
    }
    if (params.maxLGap % 2 == 0) {
        params.maxLGap++;
    }

    //call the function you want to test
    params.dest = niBlack_thresholding(params.img, params.thresh, params.minL, params.maxLGap);


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
        createTrackbar( "Thresh", imgPath, &userdata.thresh, userdata.threshMax, CallbackFunct, &userdata);
        createTrackbar( "minLenght", imgPath, &userdata.minL, userdata.minLMax, CallbackFunct, &userdata);
        createTrackbar( "maxGapLines", imgPath, &userdata.maxLGap, userdata.maxLGapMax, CallbackFunct, &userdata);

        CallbackFunct(0, &userdata);

        waitKey(0);
    }



    return 0;
}

