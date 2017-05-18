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

  // Initialize network
  LIFNetwork *network = new LIFNetwork();
  network->load_dataset(dataset.training_images, dataset.training_labels);

  // Test to see if images have loaded
  // network->show_image(network->data[0]);

  std::cout << "Initializing paramters" << '\n';
  network->initialize_params();
  std::cout << "Finished initializing paramters" << '\n';

  // Run simulation
  while(network->stime_ < 10) {
    network->cycle(true);
    std::cout << "stime_=" << network->stime_ << " firing rate=" << float(network->N_firings)/network->N << " Image: " << network->cur_img  << "\n";
    network->prepare(true);
    network->stime_++;
  }

  std::cout << "Labelling neurons.." << '\n';
  for (size_t i = 0; i < 10; i++) {
    network->labelNeurons();
  }

  std::cout << "Evaluating test set" << '\n';
  float correct = 0.;
  int size = 100;
  // int size = network->data.size();
  network->load_dataset(dataset.test_images, dataset.test_labels);
  for (size_t i = 0; i < size; i++) {
    int label = network->getLabelFromSpikes();
    std::cout << "Guessed: " << label << " versus actual: " << int(network->labels[i]) << '\n';
    if (label == int(network->labels[i])) {
      // correct!
      std::cout << "Correct!" << '\n';
      correct++;
    }
  }
  std::cout << "Accuracy: " << correct << "/" << size
            << " = " << correct / float(size) << '\n';

  return 0;
}
