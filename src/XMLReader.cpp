/**
 * @author Kabir Bertan 2122545
 */
#include <fstream>
#include <sstream>

#include "../include/XMLReader.hpp"

XMLReader::XMLReader(const std::string& filename) {
    // Open the XML file for reading
    std::ifstream xml(filename);
    if (!xml.is_open()) {
        throw std::runtime_error("Failed to open XML file: " + filename);
    }

    std::stringstream buffer;
    buffer << xml.rdbuf();            // Read the entire contents of the XML file into a string
    std::string xmlContent = buffer.str();

    size_t pos = 0;                   // Starting position for searching <space> tags
    unsigned short counter = 1;

    // Find and process all <space> in the XML file
    while ((pos = xmlContent.find("<space", pos)) != std::string::npos) {

        // Find the end of the current <space> block
        size_t endPos = xmlContent.find("</space>", pos);
        if (endPos == std::string::npos) break;

        // Extract the block of text
        std::string spaceBlock = xmlContent.substr(pos, endPos - pos);

        // Extract individual attributes from the <space> block
        int centerX = extractAttribute(spaceBlock, "center", "x");
        int centerY = extractAttribute(spaceBlock, "center", "y");
        int width = extractAttribute(spaceBlock, "size", "w");
        int height = extractAttribute(spaceBlock, "size", "h");
        int angle = extractAttribute(spaceBlock, "angle", "d");
        bool occupied = static_cast<bool>(extractAttribute(spaceBlock, "space","occupied"));

        // Create the rotated rect and the bounding box
        cv::Point center(centerX, centerY);
        cv::Size size(width, height);
        cv::RotatedRect rRect(center, size, static_cast<float>(angle));
        BoundingBox bBox(rRect, counter++, occupied);
        bBoxes.push_back(bBox);

        // Move the position forward to process the next <space> tag
        pos = endPos + 8;
    }
}


int XMLReader::extractAttribute(const std::string& block, const std::string& tag, const std::string& attribute) const {
    size_t start = block.find("<" + tag);
    start = block.find(attribute + "=\"", start);
    start += attribute.length() + 2;
    size_t end = block.find("\"", start);
    std::string valueStr = block.substr(start, end - start);

    // Convert the extracted string value to a float and cast it to an integer
    return static_cast<int>(std::stof(valueStr));
}