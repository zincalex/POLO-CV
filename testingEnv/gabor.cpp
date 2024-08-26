//
// Created by trigger on 8/21/24.
//
#include <opencv2/opencv.hpp>

using namespace cv;

// Variabili globali
int theta_slider = 45;  // Valore iniziale del trackbar (theta)
int sigma_slider = 50;  // Valore iniziale del trackbar (sigma moltiplicato per 10)
int phase_slider = 90;  // Valore iniziale del trackbar (fase in gradi)
int kernel_size = 21;
double lambda = 10.0;
double gabor_gamma = 0.5;

// Immagine originale
Mat img;

Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);

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

void updateImage(int, void*) {
    double theta = theta_slider * CV_PI / 180.0; // Converti il valore del trackbar in radianti
    double sigma = sigma_slider / 10.0; // Scala il valore di sigma
    double psi = phase_slider * CV_PI / 180.0; // Converti la fase da gradi a radianti

    // Crea il kernel di Gabor con theta, sigma e fase aggiornati
    Mat gaborKernel = getGaborKernel(Size(kernel_size, kernel_size), sigma, theta, lambda, gabor_gamma, psi, CV_32F);

    // Filtra l'immagine
    Mat filtered_img;
    filter2D(img, filtered_img, CV_32F, gaborKernel);

    // Normalizza per la visualizzazione
    normalize(filtered_img, filtered_img, 0, 255, NORM_MINMAX, CV_8U);

    // Visualizza l'immagine filtrata
    imshow("Filtered Image", filtered_img);

    Mat binary_img;
    threshold(filtered_img, binary_img, 128, 255, THRESH_BINARY);
    imshow("trh", binary_img);

    // Rilevamento delle linee con Hough Transform
    std::vector<Vec4f> lines;
    lsd->detect(filtered_img, lines);

    // Disegno delle linee rilevate sull'immagine originale
    Mat output;
    cvtColor(img, output, COLOR_GRAY2BGR);
    lsd ->drawSegments(output, lines);
    // Visualizzazione del risultato
    imshow("Detected Lines", output);
}

int main() {
    // Carica l'immagine
    img = imread("/home/trigger/Documents/GitHub/Parking_lot_occupancy/testingEnv/ParkingLot_dataset/sequence0/frames/2013-02-24_10_35_04.jpg", IMREAD_GRAYSCALE);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(4, 4));
    //clahe->apply(img, img);
    // Riduzione del rumore con GaussianBlur
    GaussianBlur(img, img, Size(5, 5), 2, 2);
    adaptiveBilateralFilter(img, 5);
    //adaptiveSharpeningByTiles(img, 5);

    cv::waitKey(0);

/*
    int kernel_size = 21;     // Dimensioni del kernel
    double theta = CV_PI / 4; // Orientazione (45 gradi)
    double gamma = 0.5;       // Rapporto d'aspetto
    double psi = CV_PI / 2;   // Fase

    // Liste per sigma e lambda a diverse scale
    std::vector<double> sigmas = {2.0, 5.0, 8.0};  // Diverse scale di sigma
    std::vector<double> lambdas = {8.0, 16.0, 24.0}; // Diverse lunghezze d'onda
*/
    /*
    // Immagine per combinare i risultati delle diverse scale
    Mat combined_result = Mat::zeros(img.size(), CV_32F);

    for (double sigma : sigmas) {
        for (double lambd : lambdas) {
            // Creazione del kernel di Gabor per ogni combinazione di sigma e lambda
            Mat gaborKernel = getGaborKernel(Size(kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, CV_32F);

            // Applicazione del filtro di Gabor
            Mat filtered_img;
            filter2D(img, filtered_img, CV_32F, gaborKernel);

            // Somma i risultati per combinare le informazioni delle diverse scale
            combined_result += filtered_img;
        }
    }

    // Normalizzazione dell'immagine combinata
    normalize(combined_result, combined_result, 0, 255, NORM_MINMAX, CV_8U);

    // Applicazione del thresholding per binarizzare l'immagine
    Mat binary_img;
    threshold(combined_result, binary_img, 128, 255, THRESH_BINARY);
    cv::imshow("binary img", binary_img);
    // Rilevamento delle linee con Hough Transform
    std::vector<Vec4i> lines;
    HoughLinesP(binary_img, lines, 1, CV_PI / 180, 50, 50, 10);

    // Disegno delle linee rilevate sull'immagine originale
    Mat output;
    cvtColor(img, output, COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(output, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Visualizzazione del risultato
    imshow("Combined Filtered Image", combined_result);
    imshow("Detected Lines", output);
     */
    namedWindow("Filtered Image", WINDOW_AUTOSIZE);
    createTrackbar("Theta", "Filtered Image", &theta_slider, 180, updateImage);
    createTrackbar("Sigma x10", "Filtered Image", &sigma_slider, 100, updateImage); // Sigma varia tra 0.0 e 10.0
    createTrackbar("Phase", "Filtered Image", &phase_slider, 360, updateImage); // La fase varia tra 0 e 360 gradi

    // Aggiorna l'immagine inizialmente
    updateImage(0, 0);
    waitKey(0);

    return 0;
}
