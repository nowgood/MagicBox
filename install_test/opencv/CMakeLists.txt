cmake_minimum_required(VERSION 3.10)
project(testopencv)

set(CMAKE_CXX_STANDARD 11)

add_executable(testopencv main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(testopencv ${OpenCV_LIBS})
