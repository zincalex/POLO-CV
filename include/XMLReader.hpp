/**
 * @author Kabir Bertan 2122545
 */
#ifndef XMLREADER_HPP
#define XMLREADER_HPP

#include <string>
#include <opencv2/imgproc.hpp>

#include "BoundingBox.hpp"


class XMLReader {
public:

    /**
     * @brief Constructor to initialize a XMLReader object for reading ground truth bounding boxes inside a xml file.
     *
     * @param xml_path    path to a xml file
     *
     * @throw std::runtime_error  if the file could not be opened
     */
    XMLReader(const std::string& xml_path);

    /**
     * @return a vector with the ground truth bounding boxes
     */
    std::vector<BoundingBox> getBBoxes() { return bBoxes; }

private:

    std::vector<BoundingBox> bBoxes;  // vector with the ground truth bounding boxes

    /**
     * @brief Extracts an integer attribute from an XML block.
     *
     *
     * @param block         XML block as a string
     * @param tag           tag name in the XML block where the attribute resides
     * @param attribute     attribute name
     *
     * @return extracted attribute value
     */
    int  extractAttribute(const std::string& block, const std::string& tag, const std::string& attribute) const;
};

#endif