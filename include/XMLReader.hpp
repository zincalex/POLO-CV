#ifndef XMLREADER_HPP
#define XMLREADER_HPP

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <string>

#include "BoundingBox.hpp"


class XMLReader {
public:
    XMLReader(const std::string& xml_path);

    std::vector<BoundingBox> getBBoxes() { return bBoxes; }

private:
    std::vector<BoundingBox> bBoxes;

    int  extractAttribute(const std::string& block, const std::string& tag, const std::string& attribute) const;
};

#endif