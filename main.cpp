/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>

// Utilities
#include "mnist/mnist_reader.hpp"

// Spiking neurons
#include "neurons/LIFNeuron.h"
// Network
#include "synapses/LIFNetwork.h"


int main(int argc, char const *argv[]) {
  // std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  // std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  // std::cout << "Test set size:      " << dataset.test_images.size() << '\n';
  // std::cout << "Image size: " << dataset.training_images[0].size() << '\n';


  // Initialize network
  LIFNetwork network;
  network.load_dataset(dataset.training_images);

  // Test to see if images have loaded
  // network.show_image(network.data[0]);

  for (size_t m = 0; m < 28; m++) {
    network.img_spikes.push_back(std::vector<unsigned>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});
  }

  // Input layer, one neuron per pixel
  for (size_t i = 0; i < dataset.training_images[0].size(); i++) {
    network.input_layer.push_back(neuron());
  }

  // Run simulation
  while(network.stime_ < 500) {
    network.cycle();
  }

  for (std::vector<unsigned> s : network.img_spikes) {
    for (unsigned k : s) {
      if (k > 0 && k < 10) {
        std::cout << k << " ";
      } else if (k >= 10) {
        std::cout << k;
      } else {
        std::cout << "0 ";
      }
    }
    std::cout << "\n";
  }
  return 0;
}
