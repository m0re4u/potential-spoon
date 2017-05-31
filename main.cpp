/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>
#include <fstream>

// Utilities
#include "mnist/mnist_reader.hpp"
#include "json/json.hpp"

// Spiking neurons
#include "neurons/LIFNeuron.h"
// Network
#include "synapses/LIFNetwork.h"

using json = nlohmann::json;

int main(int argc, char const *argv[]) {

  bool eval = true;
  std::string filename;
  if (argc == 2) {
    filename = argv[1];
  } else {
    // default configuration
    filename = "../config/isk.json";
  }

  // std::cout << "Reading in configuration: " << filename << '\n';
  // std::ifstream configFile;
  // json config;
  //
  // configFile.open(filename);
  // if (configFile.is_open()) {
  //   configFile >> config;
  // } else {
  //   std::cout << "Something went wrong while reading config(probably missing config file)" << '\n';
  // }

  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

  // Initialize network
  LIFNetwork *network = new LIFNetwork();
  network->load_dataset(dataset.training_images, dataset.training_labels);

  // Test to see if images have loaded
  // network->show_image(network->data[0]);

  std::cout << "Initializing parameters" << '\n';
  network->initialize_params();
  std::cout << "Finished initializing parameters" << '\n';

  // Run simulation
  network->t = 0 * network->ms;
  while(network->t < network->duration) {
    if (network->sleepingCycle) {
      std::cout << "t: " << network->t << " | mstime_=" << network->mstime_ << " | Image: sleeping" << "\n";
    } else {
      std::cout << "t: " << network->t << " | mstime_=" << network->mstime_ << " | Image: " << network->cur_img << "\n";
    }
    network->cycle();
    network->t += network->dt;
    network->mstime_++;
  }
  // std::cout << "Outputting training plots" << '\n';
  // network->plotSpikes();
  // network->plotNeuron();

  if (!eval) {
    return 0;
  }

  std::cout << "Resetting values" << '\n';
  network->learning = false;
  network->reset_values();

  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();

  std::cout << "Outputting labelling plots" << '\n';
  network->plotSpikes();

  std::cout << "Evaluating test set" << '\n';
  float correct = 0.;
  int size = 50;
  network->load_dataset(dataset.test_images, dataset.test_labels);

  // Per image, predict a label and check if it is correct
  for (size_t i = 0; i < size; i++) {
    int label = network->getLabelFromSpikes();
    std::cout << "Guessed: " << label << " versus actual: " << int(network->labels[i]) << '\n';
    if (label == int(network->labels[i])) {
      correct++; // correct guess
    }
  }
  std::cout << "Accuracy: " << correct << "/" << size
            << " = " << correct / float(size) << '\n';

  return 0;
}
