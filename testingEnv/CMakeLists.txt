cmake_minimum_required(VERSION 3.8)
project(testingEnv)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(BBox
        ../include/BoundingBoxes.hpp
        ../src/BoundingBoxes.cpp
)

add_executable(${PROJECT_NAME}
        trackbar_test.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} BBox)