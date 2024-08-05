#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

typedef struct ROIParams{
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
} CannyParams;

static void RoiFunct(int, void* userdata){

    CannyParams &params = *((CannyParams*)userdata);



    vector<RotatedRect> rois;
    rois.push_back(RotatedRect(Point2f(params.x_coord_center, params.y_coord_center), Size2f(params.width, params.height), params.angolo));  // Adjust the position and angle as needed
    //rois.push_back(RotatedRect(Point2f(950, 200), Size2f(200, 670), -56));  // Adjust the position and angle as needed
    //rois.push_back(RotatedRect(Point2f(900, 300), Size2f(200, 600), 30));  // Adjust the position and angle as needed

    // Draw the rotated rectangles and extract the ROIs
    params.dest = params.img.clone();
    int roiIndex = 0;
    for (const auto& roiRect : rois) {
        // Draw the rotated rectangle on the output image
        Point2f vertices[4];
        roiRect.points(vertices);
        for (int j = 0; j < 4; j++) {
            line(params.dest, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2);
        }

        // Extract and save the ROI
        getRectSubPix(params.img, roiRect.size, roiRect.center, params.roi);
        string roiName = "roi_" + to_string(roiIndex) + ".jpg";
        //imwrite(roiName, params.roi);
        roiIndex++;
    }

    // Save the output image with the rectangles
    if (!imwrite("output_with_rois.jpg", params.dest)) {
        cout << "Failed to save the image at the specified path!" << endl;
    } else {
        cout << "Image saved successfully at output_with_rois.jpg" << endl;
    }

    // Display the output image
    imshow("Output with ROIs", params.dest);

}


int main() {
    // Load the image
    string image_name = "/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence0/frames/2013-02-24_10_05_04.jpg";

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

