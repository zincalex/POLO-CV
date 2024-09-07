cmake_minimum_required(VERSION 3.8)
project(main)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(ImgProcessing
        include/ImageProcessing.hpp
        src/ImageProcessing.cpp
)

add_library(Graphics
        include/Graphics.hpp
        src/Graphics.cpp
)

add_library(BBox
        include/BoundingBox.hpp
        src/BoundingBox.cpp
)

add_library(ParkingSpaceDetector
        include/ParkingSpaceDetector.hpp
        src/ParkingSpaceDetector.cpp
)

add_library(ParkingLotStatus
        include/ParkingLotStatus.hpp
        src/ParkingLotStatus.cpp
)

add_library(Segmentation
        include/Segmentation.hpp
        src/Segmentation.cpp
)

add_library(Xml
        include/XMLReader.hpp
        src/XMLReader.cpp
)

add_library(Metrics
        include/Metrics.hpp
        src/Metrics.cpp
)

add_executable(${PROJECT_NAME}
        src/main.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ParkingSpaceDetector ParkingLotStatus Xml Metrics BBox ImgProcessing Segmentation)