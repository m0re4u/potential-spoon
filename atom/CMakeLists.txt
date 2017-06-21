# Version cmake
cmake_minimum_required (VERSION 3.5.0)

# Set nao toolchain
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/toolchain.cmake)

# Project name
project(potential-spoon)
# If not compiling: comment out the following line, and them comment back in
# after compiling & rebuild(dont ask why lol)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)

add_executable(atom RobotNetwork.cpp)
