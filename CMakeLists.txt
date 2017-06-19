cmake_minimum_required(VERSION 2.8)

project(potential_spoon C CXX)

if (CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
else ()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED "ON")
endif ()
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
add_executable(main main.cpp networks/LIFNetwork.cpp optimizations/opt1.cpp)
# CImg tester
add_executable(cimg test/CImg_test.cpp)

# Fixes pthread errors
target_link_libraries (cimg ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})
