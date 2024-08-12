//
// Created by trigger on 8/7/24.
//
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef struct Params{

    //define Mat needed to run the method to test
    Mat img;
    Mat mask;
} gParams;

Mat mergeMask(const vector<Mat>& masks){
    if(masks.empty()){
        throw runtime_error("No mask provided, check parameters");
    }


    Mat merged = masks[0].clone();

    for (size_t i = 1; i < masks.size(); ++i) {
        bitwise_or(merged, masks[i], merged);
    }

    return merged;
}

void onMouse(int event, int col, int row, int flags, void* userdata){

    Params &params = *((Params*)userdata);

    if(event == EVENT_LBUTTONDOWN)
    {
        if(params.mask.at<Vec3b>(Point(col, row))[1] == 255){
            std::cout << "Callback OK: PARKED CAR detected at pos: " << col << "   " << row << std::endl;
        } else if(params.mask.at<Vec3b>(Point(col, row))[1] == 0){
            std::cout << "Callback OK: BACKGROUND detected at pos: " << col << "   " << row << std::endl;
        }

    }
}

int main() {
    // Load the image
    Mat img = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence3/frames/2013-03-19_07_05_01.png");
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    gParams userdata;

    Mat bg, fg;
    Mat result = img.clone();

    vector<Mat> masks;

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
    Mat finalSegMask;

    int rectWidth = 170;
    int rectHeight = 100;

    int stepSizei = 170;
    int stepSizej = 100;

    while(true){
        try {
            Mat mask;
            mask.setTo(Scalar(GC_BGD));

            Rect r = selectROI(img);

            cout << "STARTED" << endl;

            grabCut(equalized_image, mask, r, bg, fg, 5, GC_INIT_WITH_RECT);

            grabCut(equalized_image, mask, r, bg, fg, 2, GC_INIT_WITH_MASK);


            Mat fgMask = Mat::zeros(img.size(), img.type());

            for (int x = 0; x < img.rows; x++) {  // Iterate over rows (height)
                for (int y = 0; y < img.cols; y++) {  // Iterate over columns (width)
                    if ((int) mask.at<uchar>(cv::Point(y, x)) == 3 || (int) mask.at<uchar>(cv::Point(y, x)) == 1) {
                        fgMask.at<Vec3b>(cv::Point(y, x))[0] = 0;
                        fgMask.at<Vec3b>(cv::Point(y, x))[1] = 255;
                        fgMask.at<Vec3b>(cv::Point(y, x))[2] = 0;
                    } else {
                        fgMask.at<Vec3b>(cv::Point(y, x))[0] = 0;
                        fgMask.at<Vec3b>(cv::Point(y, x))[1] = 0;
                        fgMask.at<Vec3b>(cv::Point(y, x))[2] = 0;
                    }
                }
            }


            if (!fgMask.empty()) {
                cout << "mask ok saving" << endl;
                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
                morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
                morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
                masks.push_back(fgMask);
            } else {
                cout << "Mask empty skipping" << endl;
            }

            finalSegMask = mergeMask(masks);
            imshow("test", finalSegMask);
        }catch(const cv::Exception& e) {
            cout << "skipping window, no foreground detected" << endl;
        }
        cout << "READY" << endl;
        int key = waitKey(0);
        if(key == 27) break;
    }

    for (int i = 0; i <= img.rows - rectHeight; i += stepSizei) {
        for (int j = 0; j <= img.cols - rectWidth ; j+= stepSizej) {
            try {
                Mat mask;
                mask.setTo(Scalar(GC_BGD));

                Rect window(j, i, rectWidth, rectHeight);

                cout << "STARTED" << endl;

                grabCut(equalized_image, mask, window, bg, fg, 5, GC_INIT_WITH_RECT);

                grabCut(equalized_image, mask, window, bg, fg, 20, GC_INIT_WITH_MASK);


                Mat fgMask = Mat::zeros(img.size(), img.type());

                for (int x = 0; x < img.rows; x++) {  // Iterate over rows (height)
                    for (int y = 0; y < img.cols; y++) {  // Iterate over columns (width)
                        if ((int) mask.at<uchar>(cv::Point(y, x)) == 3 || (int) mask.at<uchar>(cv::Point(y, x)) == 1) {
                            fgMask.at<Vec3b>(cv::Point(y, x))[0] = 0;
                            fgMask.at<Vec3b>(cv::Point(y, x))[1] = 255;
                            fgMask.at<Vec3b>(cv::Point(y, x))[2] = 0;
                        } else {
                            fgMask.at<Vec3b>(cv::Point(y, x))[0] = 0;
                            fgMask.at<Vec3b>(cv::Point(y, x))[1] = 0;
                            fgMask.at<Vec3b>(cv::Point(y, x))[2] = 0;
                        }
                    }
                }


                if (!fgMask.empty()) {
                    cout << "mask ok saving" << endl;
                    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
                    morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
                    morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
                    masks.push_back(fgMask);
                } else {
                    cout << "Mask empty skipping" << endl;
                }

                finalSegMask = mergeMask(masks);
                imshow("test", finalSegMask);
            }catch(const cv::Exception& e) {
                cout << "skipping window, no foreground detected" << endl;
            }
            cout << "READY" << endl;
            waitKey(50);
        }
    }

    addWeighted(finalSegMask, 1, result, 0.5, 0, result);
    userdata.img = result;
    userdata.mask = finalSegMask;
    imshow("output", result);
    setMouseCallback("output", onMouse, &userdata);
    waitKey(0);
    return 0;
}