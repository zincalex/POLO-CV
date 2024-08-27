#include "../include/ParkingSpaceDetector.hpp"


// Funzione per calcolare la lunghezza della linea
double calculateLineLength(const cv::Vec4i& line) {
    return std::sqrt(std::pow(line[2] - line[0], 2) + std::pow(line[3] - line[1], 2));
}

// Funzione per calcolare la distanza tra due punti
double calculateDistance(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    return std::sqrt(std::pow(pt1.x - pt2.x, 2) + std::pow(pt1.y - pt2.y, 2));
}

// Funzione per calcolare l'angolo della linea rispetto all'asse orizzontale
double calculateAngle(const cv::Vec4i& line) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];
    // line[1] - line[3] and not the opposite because the reference system for images in opencv has the y axis flipped
    return std::atan2(y2 - y1, x2 - x1) * 180 / CV_PI;
}

// Funzione per confrontare gli angoli e determinare se due linee sono simili
bool areAnglesSimilar(double angle1, double angle2, double angleThreshold = 5.0) {
    return std::abs(angle1 - angle2) < angleThreshold;
}



// Funzione per filtrare le linee duplicate
std::vector<cv::Vec4i> filterDuplicateLines(std::vector<cv::Vec4i>& lines, const cv::Mat& referenceImage, double proximityThreshold, double minLength, double angleThreshold) {
    const std::pair<double, double> FIRST_ANGLE_RANGE = std::make_pair(5.0, 20.0);
    const std::pair<double, double> SECOND_ANGLE_RANGE = std::make_pair(-87, -55.0);

    std::vector<cv::Vec4i> filteredLines;
    std::vector<bool> keepLine(lines.size(), true); // Flag per tenere traccia delle linee da mantenere

    // Sort the lines by increasing value of the x coordinate of the start point
    std::sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return a[0] < b[0];
    });

    // For each line
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!keepLine[i]) continue; // Salta la linea se è già stata scartata

        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);
        double length1 = calculateLineLength(lines[i]);
        double angle1 = calculateAngle(lines[i]);

        // 1 control : eliminate short lines
        if (length1 < minLength) {
            keepLine[i] = false; // Scarta la linea se è troppo corta
            continue;
        }


        // 2 control : mean average color around my line
        // TODO other worst version on discord kabir chat
        // TODO VISTO CHE IL CANNY CI ANDRÀ A TROVARE LE LINEE GRIGIE CREARE UN METODO A PARTE PER QUESTO CONTROLLO
        cv::Point2f center = (start1 + end1) * 0.5; // Center of the line
        cv::Size2f rectSize(length1, 3);         // Small width for the rectangle (e.g., 5 pixels)
        cv::RotatedRect rotatedRect(center, rectSize, angle1);
        cv::Rect boundingRect = rotatedRect.boundingRect();
        // Ensure the bounding rect is within the image boundaries
        boundingRect &= cv::Rect(0, 0, referenceImage.cols, referenceImage.rows);
        // Extract the region of interest (ROI)
        cv::Mat roi = referenceImage(boundingRect);
        // Create a mask for the rotated rectangle
        cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
        std::vector<cv::Point> points;
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        for (int i = 0; i < 4; ++i) {
            points.push_back(cv::Point(rectPoints[i].x - boundingRect.x, rectPoints[i].y - boundingRect.y));
        }
        cv::fillConvexPoly(mask, points, cv::Scalar(255));
        // Calculate the mean color within the masked area
        cv::Scalar meanColor = cv::mean(roi, mask);
        // Calculate the percentage of whiteness (assuming white is (255, 255, 255))
        double whiteness = (meanColor[0] + meanColor[1] + meanColor[2]) / (3.0 * 255.0);
        if (whiteness < 0.4) {
            keepLine[i] = false;
            continue;
        }


        // 3 control : eliminate lines with bad angles
        if (!((angle1 >= FIRST_ANGLE_RANGE.first && angle1 <= FIRST_ANGLE_RANGE.second) || (angle1 >= SECOND_ANGLE_RANGE.first && angle1 <= SECOND_ANGLE_RANGE.second))) {
            keepLine[i] = false;
            continue;
        }



        // Confronto con le altre
        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (!keepLine[j]) continue;

            cv::Point2f start2(lines[j][0], lines[j][1]);
            cv::Point2f end2(lines[j][2], lines[j][3]);
            double length2 = calculateLineLength(lines[j]);
            double angle2 = calculateAngle(lines[j]);

            // Calcola le distanze tra i punti iniziali e finali
            double startDistance = calculateDistance(start1, start2);
            double endDistance = calculateDistance(end1, end2);

            // 4 control : lines that start very close and end very close,  with the same angle
            if ((startDistance < proximityThreshold || endDistance < proximityThreshold) && areAnglesSimilar(angle1, angle2, angleThreshold)) {
                // Mantieni solo la linea più lunga
                keepLine[length1 >= length2 ? j : i] = false;
                if (length1 < length2)
                    break; // Se scartiamo la linea i, non ha senso confrontarla con altre linee

            }
        }
    }


    for (size_t i = 0; i < lines.size(); ++i) {
        if (!keepLine[i]) continue;
        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);
        double angle1 = calculateAngle(lines[i]);
        double length1 = calculateLineLength(lines[i]);

        for (size_t j = 0; j < lines.size(); ++j) {
            if (!keepLine[j] || i == j) continue;

            cv::Point2f start2(lines[j][0], lines[j][1]);
            cv::Point2f end2(lines[j][2], lines[j][3]);

            double startDistance = calculateDistance(start1, start2);
            double endDistance = calculateDistance(end1, end2);
            double endStartDistance = calculateDistance(end1, start2);
            double angle2 = calculateAngle(lines[j]);
            double length2 = calculateLineLength(lines[j]);
            // 5 control : if end and start are close and angle is not similar discard it
            if (startDistance <= 15 || endDistance <= 15) {
                if (!areAnglesSimilar(angle1, angle2, 20.0)) {
                    if (angle1 >= SECOND_ANGLE_RANGE.first && angle1 <= SECOND_ANGLE_RANGE.second) {
                        keepLine[j] = false;
                    }
                    else {
                        keepLine[i] = false;
                    }
                    break;
                }
            }


        }
    }


    // Raccogli le linee che sono state mantenute
    for (size_t i = 0; i < lines.size(); ++i) {
        if (keepLine[i])
            filteredLines.push_back(lines[i]);
    }

    return filteredLines;
}


std::vector<std::pair<cv::Vec4i, cv::Vec4i>> findParallelAndCloseLines(const std::vector<cv::Vec4i>& linesSupreme, cv::Mat& imagesup, double proximityThreshold, double angleThreshold) {
    std::vector<cv::Vec4i> lines = linesSupreme;
    std::sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return a[1] > b[1];  // Compare by the second element (y-coordinate of start point)
    });
    // std::vector<std::pair<cv::Vec4i, cv::Vec4i>>
    const std::pair<double, double> FIRST_ANGLE_RANGE = std::make_pair(5.0, 20.0);
    const std::pair<double, double> SECOND_ANGLE_RANGE = std::make_pair(-87, -55.0);

    std::vector<std::pair<cv::Vec4i, cv::Vec4i>> matchedLines;
    // Declare the map using the custom comparator


    // TODO make something for the palo lines
    for (size_t i = 0; i < lines.size(); ++i) {
        double angle1 = calculateAngle(lines[i]);
        cv::Point2f start1(lines[i][0], lines[i][1]);
        cv::Point2f end1(lines[i][2], lines[i][3]);

        std::optional<cv::Vec4i> bestCandidate;


        cv::Mat image = imagesup.clone();

        /*cv::line(image, cv::Point2f(start1.x, start1.y),
                 cv::Point2f(end1.x, end1.y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);*/


        double minDist = std::numeric_limits<double>::max();
        for (size_t j = i + 1; j < lines.size(); ++j) {


            double angle2 = calculateAngle(lines[j]);

            if (areAnglesSimilar(angle1, angle2, angleThreshold)) {
                cv::Point2f start2(lines[j][0], lines[j][1]);
                cv::Point2f end2(lines[j][2], lines[j][3]);
                double startDistance = calculateDistance(start1, start2);
                double startEndDistance = calculateDistance(start1, end2);
                double endStartDistance = calculateDistance(end1, start2);
                double deltaY = std::abs(start1.y - end2.y);
                double deltaX = start1.x - start2.x;

                if (startEndDistance <= 85 && (angle1 >= FIRST_ANGLE_RANGE.first && angle1 <= FIRST_ANGLE_RANGE.second) && deltaY >= 15 && startEndDistance < minDist) {
                    /*
                    cv::line(image, cv::Point2f(start2.x, start2.y),
                             cv::Point2f(end2.x, end2.y),
                             cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                    */
                    bestCandidate = lines[j];
                    minDist = startEndDistance;
                } else if (endStartDistance <= 120 && (angle1 >= SECOND_ANGLE_RANGE.first && angle1 <= SECOND_ANGLE_RANGE.second) && deltaX >= 20 && startEndDistance < minDist) {

                    /*cv::line(image, cv::Point2f(start2.x, start2.y),
                             cv::Point2f(end2.x, end2.y),
                             cv::Scalar(255, 0, 0), 2, cv::LINE_AA);*/

                    bestCandidate = lines[j];
                    minDist = startEndDistance;
                }
            }
        }
        /*cv::imshow("progress", image);
        cv::waitKey(0);*/

        if (bestCandidate.has_value()) {
            matchedLines.push_back(std::make_pair(lines[i], bestCandidate.value()));
        }

    }

    return matchedLines;
}


// Funzione per disegnare la diagonale del rotated rect e creare il rotated rect
cv::RotatedRect drawMaxDiagonalAndCreateRotatedRect(cv::Mat& image, const cv::Vec4i& line1, const cv::Vec4i& line2) {
    // Estrai i punti di inizio e fine delle due linee
    cv::Point2f start1(line1[0], line1[1]);
    cv::Point2f end1(line1[2], line1[3]);
    cv::Point2f start2(line2[0], line2[1]);
    cv::Point2f end2(line2[2], line2[3]);

    // Calcola le due distanze diagonali
    double distance1 = calculateDistance(start1, end2);
    double distance2 = calculateDistance(end1, start2);

    // Trova il massimo tra le due distanze e ottieni i punti della diagonale
    cv::Point2f diagonalStart, diagonalEnd;
    double diagonalLength;
    if (distance1 > distance2) {
        diagonalStart = start1;
        diagonalEnd = end2;
        diagonalLength = distance1;
    } else {
        diagonalStart = end1;
        diagonalEnd = start2;
        diagonalLength = distance2;
    }

    // Disegna la diagonale sull'immagine
    //cv::line(image, diagonalStart, diagonalEnd, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);  // Disegna in viola come nell'immagine

    // Calcola il centro della diagonale (centro del RotatedRect)
    cv::Point2f center = (diagonalStart + diagonalEnd) * 0.5;

    // Calcola gli angoli delle due linee parallele
    double angle1 = calculateAngle(line1);
    double angle2 = calculateAngle(line2);

    // Calcola l'angolo medio del RotatedRect
    double rotatedRectAngle = (90 + (angle1 + angle2) / 2.0);

    // Calcola l'angolo della diagonale
    double diagonalAngle = calculateAngle(cv::Vec4i(diagonalStart.x, diagonalStart.y, diagonalEnd.x, diagonalEnd.y));

    // Calcola la differenza tra l'angolo delle linee parallele e l'angolo della diagonale
    double angleDiff1 = std::abs(angle1 - diagonalAngle);
    double angleDiff2 = std::abs(angle2 - diagonalAngle);
    double averageAngleDiff = (angleDiff1 + angleDiff2) / 2.0;

    // Converti l'angolo medio in radianti
    double angleDiffRad = averageAngleDiff * CV_PI / 180.0;

    // Calcola i cateti usando trigonometria (ipotenusa è la diagonale)
    double width = std::abs(diagonalLength * std::sin(angleDiffRad));  // Larghezza è il cateto opposto
    double height = std::abs(diagonalLength * std::cos(angleDiffRad)); // Altezza è il cateto adiacente

    // Crea il RotatedRect usando il centro, la larghezza, l'altezza e l'angolo medio
    cv::RotatedRect rotatedRect(center, cv::Size2f(width, height), rotatedRectAngle);

    // Disegna il RotatedRect sull'immagine
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);  // Disegna il RotatedRect in rosso
    }

    return rotatedRect;
}


cv::Vec4i standardizeLine(const cv::Vec4i& line) {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

    // Se il punto (x2, y2) ha una coordinata x minore di (x1, y1), scambia i punti
    if (x2 < x1) {
        return cv::Vec4i(x2, y2, x1, y1);  // Ritorna la linea scambiata
    } else if (x2 == x1) {
        // Se x1 e x2 sono uguali, usa la coordinata y per determinare il punto di inizio
        if (y1 < y2) {
            // Se y1 è minore di y2, scambia i punti in modo che y1 sia quello più grande
            return cv::Vec4i(x2, y2, x1, y1);  // Ritorna la linea scambiata
        }
    }

    return line;  // Se la linea è già ordinata correttamente, non fare nulla
}


cv::Mat ParkingSpaceDetector::maskCreation(const cv::Mat& inputImg) {
    const int KERNEL_SIZE_GAUSSIAN_OTSU = 9;

    const unsigned int KERNEL_SIZE_GAUSSIAN_ADAPTIVE = 5;
    const unsigned int BLOCK_SIZE = 5;                        // Size of the pixel neighborhood used to calculate the threshold
    const unsigned int C = 2;                                 // Constant subtracted from the mean or weighted mean
    const unsigned int KERNEL_SIZE_MEDIAN_ADAPTIVE = 3;

    const double GAMMA = 2.5;
    const unsigned int SATURATION_THRESHOLD = 180;

    const unsigned int KERNEL_SIZE_CANNY = 5;
    const unsigned int LOW_THRESHOLD = 100;
    const unsigned int RATIO = 22;

    const unsigned int KERNEL_SIZE_CLOSING = 2;
    const unsigned int KERNEL_SIZE_MIN = 3;



    bool obscure = false;
    cv::Mat roiInput = ImageProcessing::createROI(inputImg, obscure);

    // CLAHE EQUALIZATION
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(roiInput, bgrChannels);
    // Crea un'istanza di CLAHE con limiti predefiniti
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(64, 64));  // Limite di contrasto 2.0, dimensione del tile 8x8
    // Applica CLAHE a ciascun canale (B, G, R)
    for (int i = 0; i < 3; i++) {
        clahe->apply(bgrChannels[i], bgrChannels[i]);
    }
    // Ricostruisci l'immagine equalizzata dai canali BGR
    cv::Mat equalizedImage;
    cv::merge(bgrChannels, equalizedImage);
    // Mostra l'immagine equalizzata
    //cv::imshow("Equalized Image", equalizedImage);
    //cv::waitKey(0);


    ////////////////////Code to threshold from roiInput, r g b components separately
    cv::Scalar lower_gray(0, 0, 0);
    cv::Scalar upper_gray(150, 150, 150);
    cv::Mat grayMask;
    cv::inRange(roiInput, lower_gray, upper_gray, grayMask);
    cv::Mat mask2;
    cv::bitwise_not(grayMask, mask2);
    cv::Mat result;
    roiInput.copyTo(result, mask2);
    //cv::imshow("Original Image", roiInput);
    //cv::imshow("Gray Mask", grayMask);
    cvtColor( result, result, cv::COLOR_BGR2GRAY );
    //cv::imshow("Result Image", result);
    //cv::waitKey(0);
    /////////////////////////////////////////


    // Adaptive mask
    cv::Mat adaptive, roiGray;
    cvtColor( equalizedImage, roiGray, cv::COLOR_BGR2GRAY );
    //cv::imshow("GrayImage", roiGray);
    //roiGray = adjustContrast(roiGray, 1.9, -130);
    //cv::imshow("CONTRASTImage", roiGray);
    GaussianBlur(roiGray, roiGray, cv::Size(KERNEL_SIZE_GAUSSIAN_ADAPTIVE,KERNEL_SIZE_GAUSSIAN_ADAPTIVE), 0,0);
    //cv::Mat bilatered;
    //cv::bilateralFilter(roiGray, bilatered, 5, 15, 5);
    //here we have to chose if using the bilatered image or the blurred one
    cv::adaptiveThreshold(roiGray, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, BLOCK_SIZE, C);
    //cv::imshow("ADAPTIVE", adaptive);
    //cv::medianBlur(adaptive, adaptive, KERNEL_SIZE_MEDIAN_ADAPTIVE);
    //cv::imshow("MEDIANBITWISENOTADAPTIVE", adaptive);


    // Otsu mask (todo USED)
    cv::Mat gray, blurred, highPass, otsuThresh;
    cvtColor(roiInput, gray, cv::COLOR_BGR2GRAY);
    gray = ImageProcessing::adjustContrast(gray, 1, -50);
    GaussianBlur(gray, blurred, cv::Size(KERNEL_SIZE_GAUSSIAN_OTSU, KERNEL_SIZE_GAUSSIAN_OTSU), 0);
    subtract(gray, blurred, highPass);  // Subtract the blurred image from the original image
    threshold(highPass, otsuThresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    //cv::imshow("FINALOTSU", otsuThresh);
    //cv::waitKey(0);


    // Saturation mask      (todo USED)
    cv::Mat gc_image = ImageProcessing::gamma_correction(roiInput, GAMMA);
    cv::Mat saturation = ImageProcessing::saturation_thresholding(gc_image, SATURATION_THRESHOLD);

    // Canny mask
    cv::Mat roiCanny;
    cvtColor( roiInput, roiGray, cv::COLOR_BGR2GRAY );
    GaussianBlur(roiGray, roiGray, cv::Size(KERNEL_SIZE_CANNY, KERNEL_SIZE_CANNY), 0);
    Canny(roiGray, roiCanny, LOW_THRESHOLD, LOW_THRESHOLD * RATIO, KERNEL_SIZE_CANNY );


    // Union of the masks
    cv::Mat mask = (otsuThresh) & saturation;
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 2)));


    // TINNING
    //mask = morphologicalSkeleton(mask);

    //cv::imshow("Resdfasfasf", mask);
    //cv::waitKey(0);
    //cv::bitwise_not(mask, mask);                      // Interested to find the areas between the lines,
    //cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    //mask = createROI(mask);                              // Adjust the white areas outside the ROI
    //cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    //cv::medianBlur(mask,mask,3);
    //cv::imshow("mask pre cont", mask);
    //cv::waitKey(0);
    /*
    // Trova i contorni
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // Crea un'immagine per visualizzare i contorni
    cv::Mat contourImage = cv::Mat::zeros(mask.size(), CV_8UC1);
    // Disegna i contorni trovati
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(contourImage, contours, (int)i, cv::Scalar(255), 2, 8, hierarchy, 0);
    }
    // Mostra il risultato
    //cv::imshow("Contours", contourImage);
    //cv::waitKey(0);


    // LSD - line detection

    cv::Mat linesDetected = roiInput.clone();
    std::vector<cv::Vec4i> lines;
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
    // LSH on gray scale of the og image
    cvtColor(roiInput, gray, cv::COLOR_BGR2GRAY);
    //lsd->detect(mask, lines);
    lsd->detect(gray, lines);
    //cv::imshow("mask", mask);
    //cv::waitKey(0);

    for (size_t i = 0; i < lines.size(); ++i)
        lines[i] = standardizeLine(lines[i]);






    // Filtra le linee duplicate in base alla vicinanza, lunghezza minima e angolo
    double proximityThreshold = 25.0; // Soglia di prossimità per considerare due linee "vicine"
    double minLength = 20;          // Lunghezza minima della linea
    double angleThreshold = 20.0;      // Soglia per considerare due angoli simili
    double max_init_distance = 120.0;  // Soglia di prossimità per considerare due linee parallele "vicine"
    double maxParallel_angle = 10.0;     // Max angle to consider two lines as parallel


    // TODO chiedere sul forum se il programma verrà testato con un numero >= di  2 foto nella sequence0
    //


    std::vector<cv::Vec4i> filteredLines = filterDuplicateLines(lines, inputImg, proximityThreshold, minLength, angleThreshold);


    cv::Mat clone = inputImg.clone();
    std::vector<std::pair<cv::Vec4i, cv::Vec4i>> parallelLines = findParallelAndCloseLines(filteredLines, clone, max_init_distance, maxParallel_angle);

    // Disegna i segmenti di linea filtrati e stampa l'angolo
    for (size_t i = 0; i < filteredLines.size(); i++) {
        cv::line(linesDetected, cv::Point2f(filteredLines[i][0], filteredLines[i][1]),
                 cv::Point2f(filteredLines[i][2], filteredLines[i][3]),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);




        //cv::circle(linesDetected, cv::Point2f(filteredLines[i][0], filteredLines[i][1]), 20, cv::Scalar(255, 0, 255), 2, cv::LINE_AA); // Anti-aliased circle
    }
    cv::imshow("Detected Lines", linesDetected);
    cv::waitKey(0);

    // Draw BB
    for (const auto& linePair : parallelLines) {
        drawMaxDiagonalAndCreateRotatedRect(linesDetected, linePair.first, linePair.second);
    }
    cv::imshow("box", linesDetected);
    cv::waitKey(0);
    */
    return mask;
}





std::vector<cv::RotatedRect> linesToRotatedRect(const std::vector<std::pair<cv::Vec4i, cv::Vec4i>>& matchedLines) {
    std::vector<cv::RotatedRect> rotatedRectCandidates;
    for (const std::pair<cv::Vec4i, cv::Vec4i>& pair : matchedLines) {
        cv::Vec4i line1 = pair.first;
        cv::Vec4i line2 = pair.second;

        // Estrai i punti di inizio e fine delle due linee
        cv::Point start1(line1[0], line1[1]);
        cv::Point end1(line1[2], line1[3]);
        cv::Point start2(line2[0], line2[1]);
        cv::Point end2(line2[2], line2[3]);

        // Calcola le due distanze diagonali
        double distance1 = calculateDistance(start1, end2);
        double distance2 = calculateDistance(end1, start2);

        // Trova il massimo tra le due distanze e ottieni i punti della diagonale
        cv::Point diagonalStart, diagonalEnd;
        double diagonalLength;
        if (distance1 > distance2) {
            diagonalStart = start1;
            diagonalEnd = end2;
            diagonalLength = distance1;
        } else {
            diagonalStart = end1;
            diagonalEnd = start2;
            diagonalLength = distance2;
        }

        // Calcola il centro della diagonale (centro del RotatedRect)
        cv::Point center = (diagonalStart + diagonalEnd) * 0.5;

        // Calcola gli angoli delle due linee parallele
        double angle1 = calculateAngle(line1);
        double angle2 = calculateAngle(line2);

        // Calcola l'angolo medio del RotatedRect
        double averageRotatedRectAngle = (90 + (angle1 + angle2) / 2.0);

        // Calcola l'angolo della diagonale
        double diagonalAngle = calculateAngle(cv::Vec4i(diagonalStart.x, diagonalStart.y, diagonalEnd.x, diagonalEnd.y));


        // Calcola la differenza tra l'angolo delle linee parallele e l'angolo della diagonale
        double angleDiff1 = std::abs(angle1 - diagonalAngle);
        double angleDiff2 = std::abs(angle2 - diagonalAngle);
        double averageAngleDiff = (angleDiff1 + angleDiff2) / 2.0;
        double angleDiffRad = averageAngleDiff * CV_PI / 180.0; // radiant conversion

        // Calcola i cateti usando trigonometria (ipotenusa è la diagonale)
        double width = std::abs(diagonalLength * std::sin(angleDiffRad));  // Larghezza è il cateto opposto
        double height = std::abs(diagonalLength * std::cos(angleDiffRad)); // Altezza è il cateto adiacente

        // Crea il RotatedRect usando il centro, la larghezza, l'altezza e l'angolo medio
        cv::RotatedRect rotatedRect(center, cv::Size2f(width, height), averageRotatedRectAngle);

        rotatedRectCandidates.push_back(rotatedRect);
    }

    return rotatedRectCandidates;
}



bool shouldKeepSmallerRect(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // Identify the smaller and larger rectangle
    cv::RotatedRect smallRect = rect1;
    cv::RotatedRect largeRect = rect2;

    if (rect1.size.area() > rect2.size.area()) {
        smallRect = rect2;
        largeRect = rect1;
    }

    // Find intersection points between the two rectangles
    std::vector<cv::Point2f> intersectionPoints;
    int result = cv::rotatedRectangleIntersection(smallRect, largeRect, intersectionPoints);

    if (result == cv::INTERSECT_FULL || result == cv::INTERSECT_PARTIAL) {
        // Calculate the intersection area (Polygon area)
        double intersectionArea = cv::contourArea(intersectionPoints);

        // Compare with the smaller rectangle's area
        double smallRectArea = smallRect.size.area();
        // If intersection area is close to the smaller rectangle's area, keep the smaller rectangle
        // Adjust threshold as needed, here it's set to 80%
        if ((intersectionArea / smallRectArea) > 0.7) {
            return true;  // Keep the smaller rectangle
        }
    }
    return false;  // No significant overlap, don't discard the larger rectangle
}

void removeOutliers(std::vector<cv::RotatedRect>& rotatedRects, const int& margin, const cv::Size& imgSize, cv::Mat& image) {

    std::vector<bool> outlier(rotatedRects.size(), false);
    const std::pair<double, double> FIRST_ANGLE_RANGE = std::make_pair(5.0, 20.0);
    const double ASPECT_RATIO_THRESHOLD = 1.4;


    // Create the mask to see where the rotated rectangle are
    cv::Mat mask = cv::Mat::zeros(imgSize, CV_8UC1);
    for (const cv::RotatedRect& rect : rotatedRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        std::vector<cv::Point> verticesVector(4);
        for (int j = 0; j < 4; j++)
            verticesVector[j] = vertices[j];
        cv::fillPoly(mask, verticesVector, cv::Scalar(255));
    }


    // Outlier elimination
    for (int i = 0; i < rotatedRects.size(); i++) {
        cv::Point2f vertices[4];
        rotatedRects[i].points(vertices);

        // Calculate the midpoints of the left and right sides
        cv::Point2f midpointLeft = (vertices[0] + vertices[3]) * 0.5;
        cv::Point2f midpointRight = (vertices[1] + vertices[2]) * 0.5;

        // Calculate the direction vectors (perpendicular to the rectangle's orientation)
        cv::Point2f direction = midpointRight - midpointLeft;
        direction = direction / cv::norm(direction);  // Normalize the direction vector

        // Calculate the endpoints for the lines
        cv::Point2f pointLeft = midpointLeft - margin * direction;
        cv::Point2f pointRight = midpointRight + margin * direction;


        bool touchedLeft = false;
        bool touchedRight = false;
        int numSteps = 50; // Number of points to check along the line
        for (int step = 1; step <= numSteps; ++step) {
            float alpha = static_cast<float>(step) / numSteps;
            cv::Point2f interpolatedPointLeft = midpointLeft * (1.0f - alpha) + pointLeft * alpha;
            cv::Point2f interpolatedPointRight = midpointRight * (1.0f - alpha) + pointRight * alpha;

            // Ensure that the points are within the image boundaries
            if (interpolatedPointLeft.x >= 0 && interpolatedPointLeft.x < imgSize.width &&
                interpolatedPointLeft.y >= 0 && interpolatedPointLeft.y < imgSize.height &&
                interpolatedPointRight.x >= 0 && interpolatedPointRight.x < imgSize.width &&
                interpolatedPointRight.y >= 0 && interpolatedPointRight.y < imgSize.height) {

                // Draw yellow dots at each interpolated point
                //cv::circle(image, interpolatedPointLeft, 3, cv::Scalar(0, 255, 255), 1); // Yellow dot
                //cv::circle(image, interpolatedPointRight, 3, cv::Scalar(0, 255, 255), 1); // Yellow dot

                // Check if the interpolated points are in the white area on the mask
                if (mask.at<uchar>(cv::Point(interpolatedPointLeft)) == 255) {
                    touchedLeft = true;
                }
                if (mask.at<uchar>(cv::Point(interpolatedPointRight)) == 255) {
                    touchedRight = true;
                }
            }
            if (touchedLeft && touchedRight) {
                outlier[i] = true;  // Mark the rectangle as an outlier
            }
            //cv::line(image, pointLeft, pointRight, cv::Scalar(0, 255, 0), 2);  // Green line with thickness 2
        }



        double angle = rotatedRects[i].angle - 90;
        float width = rotatedRects[i].size.width;
        float height = rotatedRects[i].size.height;
        float aspectRatio = std::max(width, height) / std::min(width, height);
        if ((angle >= FIRST_ANGLE_RANGE.first && angle <= FIRST_ANGLE_RANGE.second)) {
            if (aspectRatio < ASPECT_RATIO_THRESHOLD) {
                outlier[i] = true;
            }
        }
        else {
            if (aspectRatio > 1.8) {
                outlier[i] = true;
            }
        }
    }


    // Removing overlapping roteted rects
    for (int i = 0; i < rotatedRects.size(); i++) {
        if (outlier[i]) continue;
        for (int j = i + 1; j < rotatedRects.size(); j++) {
            if (outlier[j]) continue;
            int smallRectIndex = i;
            int largeRectIndex = j;

            if (rotatedRects[i].size.area() > rotatedRects[j].size.area()) {
                smallRectIndex = j;
                largeRectIndex = i;
            }

            // Find intersection points between the two rectangles
            std::vector<cv::Point2f> intersectionPoints;
            int result = cv::rotatedRectangleIntersection(rotatedRects[smallRectIndex], rotatedRects[largeRectIndex], intersectionPoints);

            if (result == cv::INTERSECT_FULL || result == cv::INTERSECT_PARTIAL) {
                // Calculate the intersection area (Polygon area)
                double intersectionArea = cv::contourArea(intersectionPoints);

                // Compare with the smaller rectangle's area
                double smallRectArea = rotatedRects[smallRectIndex].size.area();

                // If intersection area is close to the smaller rectangle's area, keep the smaller rectangle
                // Adjust threshold as needed, here it's set to 80%
                if ((intersectionArea / smallRectArea) > 0.8) {
                    outlier[largeRectIndex] = true;
                }
            }

        }
    }

    auto it = std::remove_if(rotatedRects.begin(), rotatedRects.end(),[&](const cv::RotatedRect& rect) {
                                 size_t index = &rect - &rotatedRects[0];
                                 return outlier[index];
                             });
    rotatedRects.erase(it, rotatedRects.end());

}


void drawRotatedRects(cv::Mat& image, const std::vector<cv::RotatedRect>& rotatedRects) {
    // Define the color for the border (Red)
    cv::Scalar redColor(0, 0, 255);  // BGR format, so (0, 0, 255) is red

    for (const cv::RotatedRect& rect : rotatedRects) {
        // Get the 4 vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rect.points(vertices);

        // Convert the vertices to integer points (required by polylines)
        std::vector<cv::Point> intVertices(4);
        for (int i = 0; i < 4; i++) {
            intVertices[i] = vertices[i];
        }

        // Draw the rectangle with a red border
        cv::polylines(image, intVertices, true, redColor, 1);  // Thickness of 2
    }
}




bool ParkingSpaceDetector::isWithinRadius(const cv::Point& center, const cv::Point& point, const double& radius) {
    double distance = std::sqrt(std::pow(center.x - point.x, 2) + std::pow(center.y - point.y, 2));
    return distance <= radius;
}

std::vector<cv::RotatedRect> ParkingSpaceDetector::nonMaximaSuppression(const std::vector<cv::RotatedRect>& parkingLotsBoxes, const float& iouThreshold) {
    if (parkingLotsBoxes.size() == 1) return {parkingLotsBoxes}; // Only one candidate, hence my only bounding box

    std::vector<cv::Rect> rects;
    std::vector<int> indices;


    for (const auto& entry : parkingLotsBoxes) {   // entry = (center, rect)
        rects.push_back(entry.boundingRect());
    }

    // Despite being inside the deep neural network library, the function does NOT use deep learning
    cv::dnn::NMSBoxes(rects, std::vector<float>(rects.size(), 1.0f), 0.0f, iouThreshold, indices);

    // Populate the map
    std::vector<cv::RotatedRect> validCandidates;
    for (const int& idx : indices)
        validCandidates.push_back(parkingLotsBoxes[idx]);

    return validCandidates;
}

std::vector<cv::RotatedRect> ParkingSpaceDetector::computeAverageRect(const std::vector<std::vector<cv::RotatedRect>>& boundingBoxesNMS) {
    std::vector<cv::RotatedRect> averages;

    for (const std::vector<cv::RotatedRect>& parkingSpace : boundingBoxesNMS) {
        unsigned int sumXCenter = 0, sumYCenter = 0;
        unsigned int sumWidth = 0, sumHeight = 0;
        unsigned int sumAngles = 0;
        unsigned int count = parkingSpace.size();
        float avgAngleSin = 0.0f;
        float avgAngleCos = 0.0f;

        for (const cv::RotatedRect& box : parkingSpace) {
            sumXCenter += box.center.x;
            sumYCenter += box.center.y;
            sumWidth += box.size.width;
            sumHeight += box.size.height;

            //sumAngles += box.angle;

            float angleRad = box.angle * CV_PI / 180.0f;
            avgAngleSin += std::sin(angleRad);
            avgAngleCos += std::cos(angleRad);
        }

        cv::Point avgCenter(static_cast<int>(sumXCenter / count), static_cast<int>(sumYCenter / count));
        cv::Size avgSize = cv::Size(static_cast<int>(sumWidth / count), static_cast<int>(sumHeight / count));


        //double avgAngle = sumAngles / count;
        // Calculate the average angle in radians
        float avgAngleRad = std::atan2(avgAngleSin / count, avgAngleCos / count);
        // Convert the average angle back to degrees
        float avgAngle = avgAngleRad * 180.0f / CV_PI;


        cv::RotatedRect avgRotRect (avgCenter, avgSize, avgAngle);
        averages.push_back(avgRotRect);
    }
    return averages;
}

std::vector<cv::RotatedRect> ParkingSpaceDetector::rotateBoundingBoxes(const std::vector<std::pair<cv::Point, cv::Rect>>& rects, const float& angle) {
    std::vector<cv::RotatedRect> rotatedBBoxes;
    for (const auto& pair : rects) {
        cv::Point center = pair.first;
        cv::Rect rect = pair.second;

        cv::Size size(rect.width, rect.height);
        cv::RotatedRect rotatedBBox(center, size, angle);
        rotatedBBoxes.push_back(rotatedBBox);
    }
    return rotatedBBoxes;
}


cv::Point2f getBottomRight(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    cv::Point2f bottomRight = vertices[0];
    double maxSum = bottomRight.x + bottomRight.y;
    for (int i = 1; i < 4; i++) {
        double sum = vertices[i].x + vertices[i].y;
        if (sum > maxSum) {
            maxSum = sum;
            bottomRight = vertices[i];
        }
    }
    return bottomRight;
}

void GenerateRotatedRects(std::vector<cv::RotatedRect>& rotatedRects, cv::Mat& image) {
    std::vector<cv::RotatedRect> filteredY;
    std::vector<cv::RotatedRect> filteredDegrees;
    std::vector<cv::RotatedRect> generatingRects;

    int max_X_Index = -1;
    double maxY = -1;
    double maxX = -1;

    // Loop to filter rectangles based on certain conditions
    for (size_t i = 0; i < rotatedRects.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(rotatedRects[i]);

        // Find the rectangle with the maximum Y value
        if (bottomRight.y > maxY) {
            maxY = bottomRight.y;
        }

        // Find the rectangle with the maximum X value
        if (bottomRight.x > maxX) {
            maxX = bottomRight.x;
            max_X_Index = i;
        }

        // Filter rectangles with angles between 0 and 50 degrees
        double angle = rotatedRects[i].angle;
        if (angle >= 0 && angle <= 50) {
            filteredDegrees.push_back(rotatedRects[i]);
        }

        // Filter rectangles based on Y and X values
        if (bottomRight.y > 460 && bottomRight.y < 520 && bottomRight.x > 750 && bottomRight.x < 860) {
            filteredY.push_back(rotatedRects[i]);
        }
    }

    // Remove the rectangle with the maximum Y value from the filteredY list, useful for further criterias
    for (size_t i = 0; i < filteredY.size(); ++i) {
        if (getBottomRight(filteredY[i]).y == maxY) {
            filteredY.erase(filteredY.begin() + i);
            break;
        }
    }

    // Add the rectangle with the maximum X value to the generatingRects list if it meets the condition
    if (maxX < 1250) {
        generatingRects.push_back(rotatedRects[max_X_Index]);
    }

    max_X_Index = -1;
    maxX = -1;

    // Find the rectangle with the maximum X value in filteredY
    for (size_t i = 0; i < filteredY.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(rotatedRects[i]);
        if (bottomRight.x > maxX) {
            maxX = bottomRight.x;
            max_X_Index = i;
        }
    }

    // Rectangle with the maximum X value in filteredY
    if (max_X_Index != -1) {
        generatingRects.push_back(filteredY[max_X_Index]);
    }


    double maxYValue = -1;
    int maxYIndex = -1;
    double minY_Range = image.rows + 1;
    int minY_X_Range_Index = -1;

    for (size_t i = 0; i < filteredDegrees.size(); ++i) {
        cv::Point2f bottomRight = getBottomRight(filteredDegrees[i]);

        // Find the rectangle with the maximum Y value and X > 700 with angle between 0 and 50 degree
        if (bottomRight.x > 700 && bottomRight.y > maxYValue) {
            maxYValue = bottomRight.y;
            maxYIndex = i;
        }

        // Find the rectangle with the minimum Y value and X within a specific range with angle between 0 and 50 degree
        if (bottomRight.y < minY_Range && bottomRight.x >= 685 && bottomRight.x <= 750) {
            minY_Range = bottomRight.x;
            minY_X_Range_Index = i;
        }
    }

    // Adds two "vertical" rectangles
    if (maxYIndex != -1 && minY_Range > 705) {
        generatingRects.push_back(filteredDegrees[maxYIndex]);
    }
    if (minY_X_Range_Index != -1) {
        generatingRects.push_back(filteredDegrees[minY_X_Range_Index]);
    }
    // generate new rotated rectangles
    for (const auto &rect: generatingRects) {
        cv::Point2f vertices[4];
        rect.points(vertices);

        // bottom-right Y value is less than 200
        cv::Point2f bottomRight = getBottomRight(rect);
        if (bottomRight.y < 200) {
            // Calculate the translation to align the opposite side
            cv::Point2f midPointLongSide = (vertices[0] + vertices[1]) * 0.5;
            cv::Point2f translation = bottomRight - midPointLongSide;
            cv::Point2f newCenter = rect.center - translation;
            translation.x -= cv::norm(vertices[0] - bottomRight) / 1.1;
            translation.y -= cv::norm(vertices[1] - vertices[2]) / 2;
            newCenter += translation;

            cv::RotatedRect newRect(newCenter, rect.size, rect.angle);

            rotatedRects.push_back(newRect);
        } else {
            // Y >= 200,
            cv::Point2f midPointLongSide = (vertices[0] + vertices[1]) * 0.5;
            cv::Point2f translation = bottomRight - midPointLongSide;
            cv::Point2f newCenter = rect.center + translation;

            // Create the new rotated rectangle with the same angle and size
            cv::RotatedRect newRect(newCenter, rect.size, rect.angle);

            // Check if any side is greater than 90 and resize if needed
            if (newRect.center.x > 1150) {
                if (newRect.size.width > 110) {
                    newRect.size.width *= 0.55;
                }
                if (newRect.size.height > 110) {
                    newRect.size.height *= 0.55;
                }
                // After resizing, align vertex 0 with the midpoint of the side it was projected onto
                newRect.points(vertices);
                cv::Point2f newMidPointLongSide = (vertices[0] + vertices[1]) * 0.5;
                translation = newMidPointLongSide - vertices[0];
                newCenter = newRect.center - translation;
                newRect = cv::RotatedRect(newCenter, newRect.size, newRect.angle);
            }
            rotatedRects.push_back(newRect);
        }
    }
}




float computeIoU(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // Find the intersection points between the two rotated rectangles
    std::vector<cv::Point2f> intersectionPoints;
    cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);

    if (intersectionPoints.size() <= 0)
        return 0.0f;

    // Compute the area of the intersection polygon
    float intersectionArea = cv::contourArea(intersectionPoints);

    // Calculate the area of both rectangles
    float rect1Area = rect1.size.area();
    float rect2Area = rect2.size.area();

    // Compute the IoU (Intersection over Union)
    float iou = intersectionArea / (rect1Area + rect2Area - intersectionArea);
    return iou;
}

std::vector<cv::RotatedRect> nonMaximaSuppressionROTTTT(const std::vector<cv::RotatedRect>& parkingLotsBoxes, const float& iouThreshold) {
    if (parkingLotsBoxes.size() == 1) return {parkingLotsBoxes};

    std::vector<cv::RotatedRect> validCandidates;
    std::vector<bool> suppress(parkingLotsBoxes.size(), false);

    // Apply Non-Maxima Suppression
    for (size_t i = 0; i < parkingLotsBoxes.size(); ++i) {
        if (suppress[i]) continue;

        // Keep this rectangle
        validCandidates.push_back(parkingLotsBoxes[i]);

        for (size_t j = i + 1; j < parkingLotsBoxes.size(); ++j) {
            if (suppress[j]) continue;

            // Compute IoU between the current rectangle and the other ones
            float iou = computeIoU(parkingLotsBoxes[i], parkingLotsBoxes[j]);

            // Suppress the rectangle if IoU is greater than the threshold
            if (iou < iouThreshold) {
                suppress[j] = true;
            }
        }
    }

    return validCandidates;
}



ParkingSpaceDetector::ParkingSpaceDetector(const std::filesystem::path& emptyFramesDir) {

    const double RADIUS = 40.0;
    const float IOU_THRESHOLD = 0.9;
    const float ANGLE = 10.0;

    std::vector<cv::RotatedRect> boundingBoxesCandidates;
    cv::Mat clone2;
    // Image preprocessing and find the candidate
    for (const auto& iter : std::filesystem::directory_iterator(emptyFramesDir)) {
        std::string imgPath = iter.path().string();

        // Load the image
        cv::Mat input = cv::imread(imgPath);
        if (input.empty()) {
            std::cerr << "Error opening the image" << std::endl;
        }
        cv::Size imgSize = input.size();

        //cv::Mat mask = maskCreation(input);

        // LSH line detector
        cv::Mat gray;
        std::vector<cv::Vec4i> lines;
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
        cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        lsd->detect(gray, lines);

        // Make the start and end of the lines uniform
        for (size_t i = 0; i < lines.size(); ++i)
            lines[i] = standardizeLine(lines[i]);


        // Filtra le linee duplicate in base alla vicinanza, lunghezza minima e angolo
        double proximityThreshold = 25.0; // Soglia di prossimità per considerare due linee "vicine"
        double minLength = 20;          // Lunghezza minima della linea
        double angleThreshold = 20.0;      // Soglia per considerare due angoli simili
        double max_init_distance = 120.0;  // Soglia di prossimità per considerare due linee parallele "vicine"
        double maxParallel_angle = 10.0;     // Max angle to consider two lines as parallel


        std::vector<cv::Vec4i> filteredLines = filterDuplicateLines(lines, input, proximityThreshold, minLength, angleThreshold);
        cv::Mat clone = input.clone();
        clone2 = input.clone();
        cv::Mat clone3 = input.clone();
        std::vector<std::pair<cv::Vec4i, cv::Vec4i>> parallelLines = findParallelAndCloseLines(filteredLines, clone, max_init_distance, maxParallel_angle);

        std::vector<cv::RotatedRect> rotatedRects = linesToRotatedRect(parallelLines);
        GenerateRotatedRects(rotatedRects, clone);
        //drawRotatedRects(clone, rotatedRects);
        //cv::imshow("before", clone);
        //cv::waitKey(0);

        removeOutliers(rotatedRects, 90, imgSize, clone);
        drawRotatedRects(clone, rotatedRects);

        cv::imshow("Rotated Rectangles", clone);
        cv::waitKey(0);


        boundingBoxesCandidates.insert(boundingBoxesCandidates.end(), rotatedRects.begin(), rotatedRects.end());

    }

    std::vector<std::vector<cv::RotatedRect>> boundingBoxesNonMaximaSupp;


    while (!boundingBoxesCandidates.empty()) {
        std::vector<cv::RotatedRect> parkingSpaceBoxes;

        // First populate the map with the first not analyzed parking space
        auto iterator = boundingBoxesCandidates.begin();


        cv::Point centerParkingSpace = iterator->center;
        parkingSpaceBoxes.push_back(*iterator);
        boundingBoxesCandidates.erase(iterator); // remove it in order to not insert it twice

        // Look for all the other candidates if there is one that represent the same parking lot
        auto iterator2 = boundingBoxesCandidates.begin();
        while (iterator2 != boundingBoxesCandidates.end()) {
            cv::Point anotherCenter = iterator2->center;
            if (isWithinRadius(centerParkingSpace, anotherCenter, RADIUS)) {
                parkingSpaceBoxes.push_back(*iterator);
                iterator2 = boundingBoxesCandidates.erase(iterator2);  // Erase and get the next iterator
            } else {
                ++iterator2;  // Pre-increment for efficiency purpose
            }
        }


        // All candidates for a parking space are found, need to clear them with nms
        std::vector<cv::RotatedRect> validBoxes = nonMaximaSuppressionROTTTT(parkingSpaceBoxes, IOU_THRESHOLD);


        boundingBoxesNonMaximaSupp.push_back(validBoxes);
    }

    std::vector<cv::RotatedRect> finalBoundingBoxes = computeAverageRect(boundingBoxesNonMaximaSupp);
    drawRotatedRects(clone2, finalBoundingBoxes);
    cv::imshow("Rotated Rectangles", clone2);
    cv::waitKey(0);


    unsigned short parkNumber = 1;
    for (const cv::RotatedRect rotRect : finalBoundingBoxes) {
        BoundingBox bbox = BoundingBox(rotRect, parkNumber++);
        bBoxes.push_back(bbox);
    }

}