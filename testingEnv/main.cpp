//
// Created by trigger on 8/7/24.
//
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat img = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence3/frames/2013-03-19_07_05_01.png");
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat mask, bg, fg;
    Mat result;
    Mat fgMask = Mat::zeros(img.size(), img.type());

    cv::Mat ycrcb_image;
    cvtColor(img, ycrcb_image, cv::COLOR_BGR2YCrCb);

    // Split the image into separate channels
    std::vector<cv::Mat> channels;
    split(ycrcb_image, channels);

    // Apply Histogram Equalization to the Y channel (intensity)
    equalizeHist(channels[0], channels[0]);

    // Merge the channels back
    cv::Mat equalized_image;
    merge(channels, equalized_image);

    // Convert the image back from YCrCb to BGR
    cvtColor(equalized_image, equalized_image, cv::COLOR_YCrCb2BGR);

    while (true) {
        Rect r = selectROI(img);

        grabCut(equalized_image, mask, r, bg, fg, 5, GC_INIT_WITH_RECT);

        grabCut(equalized_image, mask, r, bg, fg, 20, GC_INIT_WITH_MASK);



        for (int i = 0; i < img.rows; i++) {  // Iterate over rows (height)
            for (int j = 0; j < img.cols; j++) {  // Iterate over columns (width)
                if ((int) mask.at<uchar>(cv::Point(j, i)) == 3 || (int) mask.at<uchar>(cv::Point(j, i)) == 1) {
                    fgMask.at<Vec3b>(cv::Point(j, i))[0] = 0;
                    fgMask.at<Vec3b>(cv::Point(j, i))[1] = 255;
                    fgMask.at<Vec3b>(cv::Point(j, i))[2] = 0;
                }
            }
        }

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);

        img.copyTo(result, fgMask);

        imshow("fgMask", fgMask);
        imshow("output", result);

        int key = waitKey(0);
        if(key == 27) break;
    }
    return 0;
}