cmake_minimum_required(VERSION 2.8)

project(potential_spoon C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_COMPILER_FLAGS "-Wall")
# X11 for CImg
message(STATUS "Searching for X11...")
find_package(X11 REQUIRED)
if (X11_FOUND)
      include_directories(${X11_INCLUDE_DIR})
      link_libraries(${X11_LIBRARIES})
endif(X11_FOUND)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# pthread for CImg
message( STATUS "Searching for pthread..." )
find_package(Threads REQUIRED)

# Include libraries
include_directories(util/include)
include_directories(${CMAKE_CURRENT_LIST_DIR})

# Test a small network of LIF neurons
add_executable(LIF neurons/LIFNeuron.cpp neurons/LIFNeuron.h)
# Reading in images and present them as spike train
add_executable(main main.cpp synapses/LIFNetwork.cpp)
# CImg tester
add_executable(cimg test/CImg_test.cpp)
# SPNet
add_executable(spnet synapses/SPNet.cpp)
# add_executable(opt1 optimizations/Network.cpp)

# Fixes pthread errors
target_link_libraries (cimg ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})
