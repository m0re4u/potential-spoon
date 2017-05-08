/**
 * Main function for running an SNN on the MNIST dataset
 *
 * Author: Michiel van der Meer
 */
#include <iostream>

// Utilities
#include "mnist/mnist_reader.hpp"

// Spiking neurons
#include "neurons/LIFNeuron.h"
// Network
#include "synapses/LIFNetwork.h"


int main(int argc, char const *argv[]) {
  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  std::cout << "Test set size:      " << dataset.test_images.size() << '\n';
  std::cout << "Image size: " << dataset.training_images[0].size() << '\n';

  // Test to see if images have loaded
  // show_image(dataset.training_images[100]);

  // Initialize network
  LIFNetwork network;
  // Input layer, one neuron per pixel
  for (size_t i = 0; i < dataset.training_images[0].size(); i++) {
    network.input_layer.push_back(neuron());
  }

  while(network.stime_ < 1000) {
    network.cycle();
  }

  return 0;
}
