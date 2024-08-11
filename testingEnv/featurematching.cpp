//
// Created by trigger on 8/11/24.
//
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // Load the Haar Cascade file for face detection
    CascadeClassifier face_cascade;
    face_cascade.load("paltes.xml");

    // Load the image
    Mat img = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence2/frames/2013-03-09_09_30_04.png");
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

    // Draw bounding boxes around detected faces
    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(img, faces[i], Scalar(255, 0, 0), 2);
    }

    // Display the result
    imshow("Detected Faces", img);
    waitKey(0);
    return 0;
}