/**
 * @author Alessandro Viespoli 2120824
 */
#ifndef PARKINGLOTSTATUS_HPP
#define PARKINGLOTSTATUS_HPP

#include <opencv2/imgproc.hpp>

#include "BoundingBox.hpp"


class ParkingLotStatus {
public:

    /**
     * @brief Constructor to initialize a ParkingLotStatus object. For each bounding box, the method look at the parking image
     *        and based on a cascade of controls it determine if a parking space is occupied or not. The cascade of controls are :
     *           1. white pixel percentage in the HSV binary segmentation mask.
     *           2. white pixel percentage for saturation thresholding on the HSV version of the image.
     *           3. white pixel percentage for thresholding darker areas in the RGB image.
     *           4. control the number of features by applying the filtering stage of meanshift segmentation and then detect features with SIFT.
     *
     * @param parkingImage           image from which detect the cars
     * @param bBoxes                 bounding boxes of where the parking spaces are
     * @param segmentationMaskHSV    segmentation hsv mask of the image, needed for car detection
     */
    ParkingLotStatus(const cv::Mat& parkingImage, std::vector<BoundingBox> bBoxes, const cv::Mat& segmentationMaskHSV);

    /**
     * @brief Visualizes the status of the parking lot by drawing bounding boxes and parking space numbers on the parking image.
     *        An empty parking lot is blue colored, while an occupied parking space is red colored.
     *
     * @return an image with the status
     */
    cv::Mat seeParkingLotStatus();

    /**
     * @return a list of parking space numbers that are currently occupied
     */
    std::vector<unsigned short> getOccupiedParkingSpaces() const;

    /**
     * @return a vector with the updated bounding boxes
     */
    std::vector<BoundingBox> getStatusPredictions() const { return bBoxes; }


private:
    cv::Mat parkingImage;                // image of the parking lot
    std::vector<BoundingBox> bBoxes;     // Bounding boxes that represent the parking spaces

    /**
     * @brief Determines if a parking space contains a car based on the percentage of white pixels in a mask.
     *
     * @param mask              binary mask representing a parking space
     * @param totalPixels       total number of pixels in the mask
     * @param percentage        threshold percentage of white pixels needed to classify the parking space as occupied.
     *
     * @return true if the percentage of white pixels in the mask is greater or equal than `percentage`, false otherwise
     */
    bool isCar(const cv::Mat& mask, const int& totalPixels, const double& percentage) const;
};

#endif