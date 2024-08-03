cmake_minimum_required(VERSION 3.8)
project(main)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(BBox
        include/BoundingBox.hpp
        src/BoundingBox.cpp
)

add_executable(${PROJECT_NAME}
        src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} BBox)