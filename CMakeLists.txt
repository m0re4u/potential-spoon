cmake_minimum_required(VERSION 2.8)

project(potential_spoon)

include_directories(/home/m0re/projects/uni/potential-spoon/eigen)

add_executable(neuron neuron.cpp)
add_executable(network network.cpp)
