CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(eigenface)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "/home/lijiachen/opencv-3.4.2/release/share/OpenCV")
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mytrain mytrain.cpp utils.cpp)
add_executable(mytest mytest.cpp utils.cpp)
target_link_libraries(mytrain ${OpenCV_LIBS})
target_link_libraries(mytest ${OpenCV_LIBS})
