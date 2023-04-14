cmake_minimum_required(VERSION 2.8.12)

PROJECT(YOLOv5)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(caging caging.cpp src/Yolov5ObjectDetector.cpp src/utils.cpp)
TARGET_INCLUDE_DIRECTORIES(caging PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(caging ${OpenCV_LIBS})


