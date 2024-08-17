#include "../include/XMLReader.hpp"

// Function to extract attributes of each BB from xml file
int XMLReader::extractAttribute(const std::string& block, const std::string& tag, const std::string& attribute) const {
    size_t start = block.find("<" + tag);
    start = block.find(attribute + "=\"", start);
    start += attribute.length() + 2;
    size_t end = block.find("\"", start);
    std::string valueStr = block.substr(start, end - start);

    return static_cast<int>(std::stof(valueStr));
}


XMLReader::XMLReader(const std::string& filename) {

    std::ifstream xml(filename);
    if (!xml.is_open()) {
        throw std::runtime_error("Failed to open XML file: " + filename);
    }

    std::stringstream buffer;
    buffer << xml.rdbuf();
    std::string xmlContent = buffer.str();

    size_t pos = 0;
    unsigned short counter = 1;

    // Find and process all <space> in the XML file
    while ((pos = xmlContent.find("<space", pos)) != std::string::npos) {
        size_t endPos = xmlContent.find("</space>", pos);
        if (endPos == std::string::npos) break;

        std::string spaceBlock = xmlContent.substr(pos, endPos - pos);


        int centerX = extractAttribute(spaceBlock, "center", "x");
        int centerY = extractAttribute(spaceBlock, "center", "y");
        int width = extractAttribute(spaceBlock, "size", "w");
        int height = extractAttribute(spaceBlock, "size", "h");
        int angle = extractAttribute(spaceBlock, "angle", "d");
        bool occupied = static_cast<bool>(extractAttribute(spaceBlock, "space","occupied"));

        cv::Point center(centerX, centerY);
        cv::Size size(width, height);
        cv::RotatedRect rRect(center, size, static_cast<float>(angle));

        BoundingBox bBox(rRect, counter++, occupied);
        bBoxes.push_back(bBox);

        // Next BB
        pos = endPos + 8;
    }

}
