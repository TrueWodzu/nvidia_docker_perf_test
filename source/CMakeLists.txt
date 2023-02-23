cmake_minimum_required(VERSION 3.16)
project(perftest VERSION 0.0.1 DESCRIPTION "GPU performance test.")
set(CMAKE_CXX_FLAGS "-fopenmp -fPIC")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(perftest
  main.cpp
)

target_compile_features(perftest PRIVATE cxx_std_17)

target_include_directories(perftest PRIVATE /usr/local/include/opencv4)
target_include_directories(perftest PRIVATE /usr/local/lib)

target_link_libraries(perftest /usr/local/lib/libopencv_core.so)
target_link_libraries(perftest /usr/local/lib/libopencv_dnn.so)

include(GNUInstallDirs)
