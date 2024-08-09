//
// Created by trigger on 8/6/24.
//


#include <opencv2/opencv.hpp>

int main() {
    // Carica l'immagine
    cv::Mat src = cv::imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/ParkingLot_dataset/sequence0/frames/2013-02-24_10_35_04.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Errore nel caricare l'immagine!" << std::endl;
        return -1;
    }

    // Converti l'immagine in scala di grigi
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Applica il filtro Sobel per rilevare i bordi
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::Mat edges;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    // Inverti i colori per ottenere un effetto simile
    cv::Mat edges_inv  = edges;
    //cv::bitwise_not(edges, edges_inv);

    // Converti in BGR per visualizzare
    cv::Mat edges_inv_bgr;
    cv::cvtColor(edges_inv, edges_inv_bgr, cv::COLOR_GRAY2BGR);

    // Visualizza le immagini
    cv::imshow("Originale", src);
    cv::imshow("Bordi Invertiti", edges_inv_bgr);

    // Salva l'immagine risultante
    cv::imwrite("edges_inverted.png", edges_inv_bgr);

    cv::waitKey(0);
    return 0;
}