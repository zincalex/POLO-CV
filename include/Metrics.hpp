#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include "../include/BoundingBox.hpp"

class Metrics {
public:
    Metrics(const std::vector<BoundingBox>& groundTruth, const std::vector<BoundingBox>& bBoxesPrediction);

    double computeMeanAveragePrecision() const;

private:


    int totBoundingBoxes;
};

#endif