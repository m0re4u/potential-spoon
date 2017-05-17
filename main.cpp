/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>

// Utilities
#include "mnist/mnist_reader.hpp"
#include "json/json.hpp"

// Spiking neurons
#include "neurons/LIFNeuron.h"
// Network
#include "synapses/LIFNetwork.h"

using json = nlohmann::json;

int main(int argc, char const *argv[]) {
  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  // std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  // std::cout << "Test set size:      " << dataset.test_images.size() << '\n';
  // std::cout << "Image size: " << dataset.training_images[0].size() << '\n';


  // Initialize network
  LIFNetwork *network = new LIFNetwork();
  network->load_dataset(dataset.training_images);

  // Test to see if images have loaded
  // network->show_image(network->data[0]);

  network->initialize_params();

  // Run simulation
  while(network->stime_ < 100) {
    network->cycle();
    std::cout << "stime_=" << network->stime_ << " firing rate=" << float(network->N_firings)/network->N << " Image: " << network->cur_img  << "\n";
    network->prepare();
    network->stime_++;
  }

  return 0;
}
