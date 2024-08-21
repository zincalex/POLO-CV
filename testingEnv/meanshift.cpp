//
// Created by trigger on 8/20/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>


// Funzione per applicare lo sharpening adattivo basato sul contenuto di ciascun tile
void adaptiveSharpeningByTiles(cv::Mat& image, int tileSize) {
    // Controlla che l'immagine sia in scala di grigi
    if (image.channels() != 1) {
        std::cerr << "L'immagine deve essere in scala di grigi!" << std::endl;
        return;
    }

    // Definiamo diversi kernel di sharpening per i 5 stadi
    std::vector<cv::Mat> sharpeningKernels(5);

    // Sharpening leggero
    sharpeningKernels[0] = (cv::Mat_<float>(3, 3) <<
            0, -0.2, 0,
            -0.2, 1.8, -0.2,
            0, -0.2, 0);

    // Sharpening moderato
    sharpeningKernels[1] = (cv::Mat_<float>(3, 3) <<
            0, -0.5, 0,
            -0.5, 3, -0.5,
            0, -0.5, 0);

    // Sharpening normale
    sharpeningKernels[2] = (cv::Mat_<float>(3, 3) <<
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0);

    // Sharpening forte
    sharpeningKernels[3] = (cv::Mat_<float>(3, 3) <<
            0, -1.5, 0,
            -1.5, 7, -1.5,
            0, -1.5, 0);

    // Sharpening molto forte
    sharpeningKernels[4] = (cv::Mat_<float>(3, 3) <<
            0, -2, 0,
            -2, 9, -2,
            0, -2, 0);

    // Suddividi l'immagine in tiles
    int rows = image.rows;
    int cols = image.cols;

    for (int y = 0; y < rows; y += tileSize) {
        for (int x = 0; x < cols; x += tileSize) {
            // Definisci il rettangolo del tile
            int tileWidth = std::min(tileSize, cols - x);
            int tileHeight = std::min(tileSize, rows - y);
            cv::Rect tileRect(x, y, tileWidth, tileHeight);

            // Estrai il tile dall'immagine
            cv::Mat tile = image(tileRect);

            // Calcola la varianza locale del tile per determinare la quantità di dettagli
            cv::Scalar mean, stddev;
            cv::meanStdDev(tile, mean, stddev);
            double variance = stddev[0] * stddev[0];

            // Determina il livello di sharpening in base alla varianza
            int sharpeningLevel;
            if (variance < 100) {
                sharpeningLevel = 0;  // Sharpening leggero
            } else if (variance < 300) {
                sharpeningLevel = 1;  // Sharpening moderato
            } else if (variance < 600) {
                sharpeningLevel = 2;  // Sharpening normale
            } else if (variance < 1000) {
                sharpeningLevel = 3;  // Sharpening forte
            } else {
                sharpeningLevel = 4;  // Sharpening molto forte
            }

            // Applica il kernel di sharpening corrispondente al livello
            cv::filter2D(tile, tile, -1, sharpeningKernels[sharpeningLevel]);
        }
    }
}


// Funzione per applicare il filtro bilaterale adattivo basato sul contenuto di ciascun tile
void adaptiveBilateralFilter(cv::Mat& image, int tileSize) {
    // Assicurati che l'immagine sia del tipo CV_8UC1 o CV_8UC3
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        std::cerr << "L'immagine deve essere in scala di grigi (CV_8UC1) o a colori (CV_8UC3)!" << std::endl;
        return;
    }

    // Crea una maschera vuota per l'immagine filtrata
    cv::Mat filteredImage = cv::Mat::zeros(image.size(), image.type());

    // Suddividi l'immagine in tiles
    int rows = image.rows;
    int cols = image.cols;

    for (int y = 0; y < rows; y += tileSize) {
        for (int x = 0; x < cols; x += tileSize) {
            // Definisci il rettangolo del tile
            int tileWidth = std::min(tileSize, cols - x);
            int tileHeight = std::min(tileSize, rows - y);
            cv::Rect tileRect(x, y, tileWidth, tileHeight);

            // Estrai il tile dall'immagine
            cv::Mat tile = image(tileRect);

            // Crea una copia del tile per l'elaborazione
            cv::Mat tileCopy;
            tile.copyTo(tileCopy);

            // Calcola la varianza locale del tile per determinare la quantità di dettagli
            cv::Scalar mean, stddev;
            cv::meanStdDev(tileCopy, mean, stddev);
            double variance = stddev[0] * stddev[0];

            // Adatta i parametri del filtro bilaterale in base alla varianza
            int d = 9;  // Diametro del filtro
            double sigmaColor, sigmaSpace;

            if (variance < 300) {
                // Aree piatte: applica un forte smoothing
                sigmaColor = 50;
                sigmaSpace = 50;
            } else if (variance < 500) {
                // Aree con qualche dettaglio: smoothing moderato
                sigmaColor = 25;
                sigmaSpace = 25;
            } else {
                // Aree dettagliate: smoothing leggero per preservare i bordi
                sigmaColor = 10;
                sigmaSpace = 10;
            }

            // Applica il filtro bilaterale adattivo sulla copia del tile
            cv::Mat tileFiltered;
            cv::bilateralFilter(tileCopy, tileFiltered, d, sigmaColor, sigmaSpace);

            // Copia il risultato filtrato nel tile originale (nella maschera)
            tileFiltered.copyTo(filteredImage(tileRect));
        }
    }

    // Sostituisci l'immagine originale con quella filtrata
    image = filteredImage.clone();  // Usa clone per assicurarti che i dati siano separati
}
int main() {
    // Carica l'immagine
    std::string img_path = "/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence0/frames/2013-02-24_10_35_04.jpg";  // Sostituisci con il percorso della tua immagine
    cv::Mat image = cv::imread(img_path);
    cv::imshow("src", image);
    if (image.empty()) {
        std::cerr << "Immagine non trovata!" << std::endl;
        return -1;
    }

    // Suddividi l'immagine nei suoi canali BGR
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(image, bgrChannels);

    // Crea un'istanza di CLAHE con limiti predefiniti
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(4, 4));  // Limite di contrasto 2.0, dimensione del tile 8x8

    // Applica CLAHE a ciascun canale (B, G, R)
    for (const cv::Mat& channel : bgrChannels) {
        clahe->apply(channel, channel);
    }

    // Ricostruisci l'immagine equalizzata dai canali BGR
    cv::Mat equalizedImage;
    cv::merge(bgrChannels, equalizedImage);

    // Mostra l'immagine equalizzata
    cv::imshow("Equalized Image", equalizedImage);


    //cv::cvtColor(equalizedImage, equalizedImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> bgrChannels_bilateral(3);
    cv::Mat blurred, highPass;
    for (int i = 0; i < 3; i++) {
        cv::bilateralFilter(bgrChannels[i], bgrChannels_bilateral[i], 9, 25,25);
        cv::imshow("bilateral_channel", bgrChannels_bilateral[i]);
        cv::waitKey(0);
    }
    cv::merge(bgrChannels_bilateral, blurred);

    cv::Mat meanShiftResult;
    int spatialRadius = 10;  // Raggio spaziale (spatial window radius)
    int colorRadius = 50;    // Raggio di colore (color window radius)
    cv::pyrMeanShiftFiltering(blurred, meanShiftResult, spatialRadius, colorRadius);

    // Mostra il risultato dell'algoritmo Mean Shift
    cv::imshow("Mean Shift Result", meanShiftResult);


    cv::imshow("blurred", blurred);
    //cv::GaussianBlur(equalizedImage, blurred, cv::Size(5,5), 0, 0);
    subtract(equalizedImage, meanShiftResult, highPass);

    std::vector<cv::Mat> bgrChannelsMeanShift;
    cv::split(highPass, bgrChannelsMeanShift);
    cv::Mat otsu_img;
    std::vector<cv::Mat>otsu_channels;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    for (const cv::Mat& channel : bgrChannelsMeanShift) {
        cv::threshold(channel, channel, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        //cv::morphologyEx(channel, channel, cv::MORPH_ERODE, kernel);
        cv::imshow("OTSU_channel", channel);
        cv::waitKey(0);
        otsu_channels.push_back(channel);
    }

    cv::merge(otsu_channels, otsu_img);



    cv::imshow("OTSU", otsu_img);




    // Applica l'algoritmo Canny per rilevare i bordi
    cv::Mat edges;
    cv::Canny(otsu_img, edges, 100, 200);
    // Trova i contorni nell'immagine dei bordi
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Crea un'immagine per visualizzare i contorni
    cv::Mat contourImage = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2);  // Disegna i contorni in verde

    // Mostra l'immagine con i contorni rilevati
    cv::imshow("Contours", contourImage);

    // Mostra l'immagine dei bordi rilevati con Canny
    cv::imshow("Canny Edges", edges);

    // Attendi la pressione di un tasto
    cv::waitKey(0);

    return 0;
}