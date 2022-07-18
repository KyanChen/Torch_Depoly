cmake_minimum_required(VERSION 3.0.0)
project(DeployCpp VERSION 1.0.0)
# set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE=Release)
# include(CTest)
# enable_testing()
set(Torch_DIR /Users/chenkeyan/libs/libtorch/share/cmake/Torch)
set(OpenCV_DIR /Users/chenkeyan/libs/opencv_4_6_0/lib/cmake/opencv4)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV library status:")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")

add_executable(DeployCpp LibTorch.cpp)

target_link_libraries(DeployCpp ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
set_property(TARGET DeployCpp PROPERTY CXX_STANDARD 14)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
