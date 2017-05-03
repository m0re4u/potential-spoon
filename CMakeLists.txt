cmake_minimum_required(VERSION 2.8)

project(potential_spoon CXX)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

include_directories(/home/m0re/projects/uni/potential-spoon/eigen)

add_executable(neuron neurons/neuron.cpp)
add_executable(sample neurons/test.cpp)
add_executable(brain synapses/brain_syn.cpp)
