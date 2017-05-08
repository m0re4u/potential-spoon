cmake_minimum_required(VERSION 2.8)

project(potential_spoon C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# X11 for CImg
message(STATUS "Searching for X11...")
find_package(X11 REQUIRED)
IF (X11_FOUND)
      include_directories(${X11_INCLUDE_DIR})
      link_libraries(${X11_LIBRARIES})
ENDIF (X11_FOUND)
# pthread for CImg
message( STATUS "Searching for pthread..." )
find_package(Threads REQUIRED)

# Include libraries
include_directories(util/include)

# Test a small network of LIF neurons
add_executable(LIF synapses/LIFNeuron.cpp synapses/LIFNeuron.h)
# Reading in images and present them as spike train
add_executable(main main.cpp)
# CImg tester
add_executable(cimg test/CImg_test.cpp)

# Fixes pthread errors
target_link_libraries (cimg ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})
