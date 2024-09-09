//
// Created by trigger on 9/8/24.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;


cv::Mat smallContoursElimination(const cv::Mat& input_mask) {
    cv::Mat in_mask;
    in_mask = input_mask.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(in_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// Filtro dei contorni per dimensione
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        // Se il contorno è troppo piccolo, lo eliminiamo
        if (area < 500) { // Modifica la soglia a seconda delle dimensioni del tuo dataset
            cv::drawContours(in_mask, contours, static_cast<int>(i), cv::Scalar(0), cv::FILLED);
        }
    }
    return in_mask;
}

cv::Mat grabCutMask(const cv::Mat& input_mask, const cv::Mat& input_img) {
    cv::Mat grabcut_mask = input_mask.clone();
    cv::Mat start = input_img.clone();
    cv::Mat grabcut_partial;

    grabcut_mask.setTo(255, grabcut_mask = 127);

    grabcut_mask.setTo(cv::GC_BGD, grabcut_mask == 0);
    grabcut_mask.setTo(cv::GC_PR_FGD, grabcut_mask == 255);


    cv::Mat bgdModel, fgdModel;

    std::cout << "Starting GrabCut..." << std::endl;
    try {
        cv::grabCut(start, grabcut_mask, cv::Rect(), bgdModel, fgdModel, 30, cv::GC_INIT_WITH_MASK);
        start.copyTo(grabcut_partial, grabcut_mask);
        cv::grabCut(grabcut_partial, grabcut_mask, cv::Rect(), bgdModel, fgdModel, 30, cv::GC_INIT_WITH_MASK);
    }
    catch (const cv::Exception &e) {
        std::cout << "no foreground detected" << std::endl;
    }
    std::cout << "GrabCut OK" << std::endl;

    cv::compare(grabcut_mask, cv::GC_PR_FGD, grabcut_mask, cv::CMP_EQ);
    // Filtro dei contorni per dimensione
    cv::morphologyEx(grabcut_mask, grabcut_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    cv::morphologyEx(grabcut_mask, grabcut_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

    grabcut_mask = smallContoursElimination(grabcut_mask);

    cv::Mat fgMask = cv::Mat::zeros(start.size(), CV_8UC3);
    for (int x = 0; x < start.rows; x++) {  // Iterate over rows (height)
        for (int y = 0; y < start.cols; y++) {  // Iterate over columns (width)
            if ((int) grabcut_mask.at<uchar>(cv::Point(y, x)) == 255) {
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[0] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[1] = 255;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[2] = 0;
            } else {
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[0] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[1] = 0;
                fgMask.at<cv::Vec3b>(cv::Point(y, x))[2] = 0;
            }
        }
    }
    return fgMask;
}



// Funzione per caricare le immagini di background e addestrare il modello MOG2
Ptr<BackgroundSubtractorMOG2> trainBackgroundModel(const vector<String>& backgroundImages) {
    // Crea il sottrattore di background MOG2
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();

    for (const auto& imagePath : backgroundImages) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Errore nel caricamento dell'immagine: " << imagePath << endl;
            continue;
        }

        // Applica il modello di background all'immagine di background per "allenarlo"
        Mat fgMask;
        mog2->apply(image, fgMask, 0.55);  // Usa un learning rate basso per adattarsi lentamente
    }

    return mog2;  // Ritorna il modello di background addestrato
}

// Funzione per applicare il modello su una nuova immagine con auto e applicare la maschera
void detectForeground(Ptr<BackgroundSubtractorMOG2>& mog2, const vector<String>& imagesWithCars) {
    for (const auto& imagePath : imagesWithCars) {
        Mat imageWithCars = imread(imagePath);
        if (imageWithCars.empty()) {
            cerr << "Errore: immagine con auto non trovata: " << imagePath << endl;
            continue;
        }

        // Applica il modello di background addestrato all'immagine con le auto
        Mat fgMask;
        mog2->apply(imageWithCars, fgMask, 0);  // Usa un learning rate di 0 per non aggiornare piÃ¹ il background

        fgMask = smallContoursElimination(fgMask);

        for (int x = 0; x < fgMask.rows; x++) {  // Iterate over rows (height)
            for (int y = 0; y < fgMask.cols; y++) {  // Iterate over columns (width)
                if ((unsigned char) fgMask.at<uchar>(cv::Point(y, x)) == 127) {
                    fgMask.at<uchar>(cv::Point(y, x)) = 0;
                }
            }
        }

        fgMask = smallContoursElimination(fgMask);

        //cv::morphologyEx(fgMask, fgMask, cv::MORPH_DILATE,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20)));

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// Crea un'immagine su cui disegnare
        cv::Mat drawing = cv::Mat::zeros(fgMask.size(), CV_8UC1);

// Per ogni contorno trovato, riempi il poligono se è convesso
        for (size_t i = 0; i < contours.size(); i++) {
                cv::fillPoly(drawing, contours[i], cv::Scalar(255, 255, 255)); // Riempi con colore verde
        }



        imshow("fgMask post for", drawing);
        //cv::waitKey(0);




        // Crea una copia dell'immagine originale
        Mat maskedImage = imageWithCars.clone();

        grabCutMask(fgMask,maskedImage);

        imshow("fgmask", fgMask);

        cv::imshow("grabcut", fgMask);

        // Imposta a nero il background dove la maschera ha valore 0
        maskedImage.setTo(Scalar(0, 0, 0), drawing == 0);  // Il background diventa nero




        // Mostra l'immagine originale, la maschera e l'immagine con il background nero
        imshow("Original Image with Cars", imageWithCars);

        //imshow("Foreground Mask (Cars Detected)", fgMask);
        imshow("Image with Background Removed", maskedImage);

        // Attendi che l'utente prema un tasto prima di passare alla prossima immagine
        waitKey(0);
    }
}

int main() {
    // Step 1: Carica le immagini di background (le 70 immagini senza auto)
    vector<String> backgroundImages;
    glob("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence0/frames", backgroundImages);  // Carica tutte le immagini nella cartella

    if (backgroundImages.empty()) {
        cerr << "Errore: nessuna immagine di background trovata!" << endl;
        return -1;
    }

    // Step 2: Addestra il modello di background con le immagini di background
    Ptr<BackgroundSubtractorMOG2> mog2 = trainBackgroundModel(backgroundImages);
    cout << "Modello di background addestrato!" << endl;

    // Step 3: Carica tutte le immagini della cartella sequence4 con le auto
    vector<String> imagesWithCars;
    glob("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence1/frames/*", imagesWithCars);

    if (imagesWithCars.empty()) {
        cerr << "Errore: nessuna immagine con auto trovata!" << endl;
        return -1;
    }

    // Step 4: Rileva le auto usando il modello di background addestrato su tutte le immagini di sequence4
    detectForeground(mog2, imagesWithCars);


    destroyAllWindows();

    return 0;
}