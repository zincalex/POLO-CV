//
// Created by trigger on 8/18/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Vec3b getRandomColor() {
    return Vec3b(rand() % 256, rand() % 256, rand() % 256);
}

int main() {
    // 1. Carica l'immagine
    Mat img = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence4/frames/2013-04-15_07_15_01.png");
    if (img.empty()) {
        cout << "Errore nel caricamento dell'immagine!" << endl;
        return -1;
    }



    // 2. Converti l'immagine in spazio colore Lab (più appropriato per clustering)
    Mat imgLab;
    cvtColor(img, imgLab, COLOR_BGR2Lab);

    // 3. Reshape l'immagine a un array di punti 2D (ogni pixel sarà un punto nel nostro spazio dei colori)
    Mat imgReshaped = imgLab.reshape(1, imgLab.total());

    // 4. Converti in float per kmeans
    imgReshaped.convertTo(imgReshaped, CV_32F);

    // 5. Definisci i parametri di K-means
    int k = 3;  // Numero di cluster (puoi sperimentare con diversi valori)
    Mat labels;
    Mat centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 300, 0.2);

    // 6. Applica K-means
    kmeans(imgReshaped, k, labels, criteria, 3, KMEANS_RANDOM_CENTERS, centers);

    vector<Mat> clusterImages(k);

    for (int i = 0; i < k; i++) {
        // Crea un'immagine nera di base
        clusterImages[i] = Mat::zeros(img.size(), img.type());
    }

    // 8. Assegna i pixel ai rispettivi cluster
    for (int i = 0; i < img.total(); i++) {
        int cluster_idx = labels.at<int>(i);
        // Copia il pixel dell'immagine originale nel rispettivo cluster
        clusterImages[cluster_idx].at<Vec3b>(i) = img.at<Vec3b>(i);
    }

    // 9. Visualizza ogni cluster in una finestra separata
    for (int i = 0; i < k; i++) {
        string windowName = "Cluster " + to_string(i);
        imshow(windowName, clusterImages[i]);
    }

    // 10. Visualizza l'immagine originale per riferimento
    imshow("Immagine Originale", img);

    waitKey(0);
    return 0;
}