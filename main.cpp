/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>
// time
#include <chrono>

// Utilities
#include "mnist/mnist_reader.hpp"

// Spiking neurons
#include "neurons/LIFNeuron.h"
// Network
#include "synapses/LIFNetwork.h"

int main(int argc, char const *argv[]) {

  bool eval = true;

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

  network->showWeightExtrema();

  // std::chrono::time_point<std::chrono::system_clock> start, end;
  // Run simulation
  while(network->cur_img < network->train_limit) {
    network->cycle();
    network->t += network->dt;
    network->mstime_++;
    // std::cout << "Time: " << network->t << " current image: " << network->cur_img << '\n';
    std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
              << (network->cur_img / float(network->train_limit))<< std::flush;
  }
  std::cout << '\n';

  std::cout << "Outputting training plots" << '\n';
  network->plotSpikes();
  // network->plotNeuron();
  // network->plotWeights();

  network->saveWeights();

  network->showWeightExtrema();

  if (!eval) {
    return 0;
  }

  std::cout << "Resetting values" << '\n';
  network->learning = false;
  network->reset_values();

  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();

  // std::cout << "Outputting labelling plots" << '\n';
  // network->plotSpikes();

  std::cout << "Evaluating test set" << '\n';
  float correct = 0.;
  network->load_dataset(dataset.test_images, dataset.test_labels);

  // Per image, predict a label and check if it is correct
  int i = 0;
  while (network->cur_img < network->test_limit) {
    // start = std::chrono::system_clock::now();
    int label = network->getLabelFromSpikes();
    // end = std::chrono::system_clock::now();
    // std::chrono::duration<double> dur = end - start;
    // std::cerr << "Classification time: " << dur.count() << "s\n";

    std::cout << "Index: " << i << " Guessed: " << label << " versus actual: " << int(network->labels[i]) << '\n';
    if (label == int(network->labels[i])) {
      correct++; // correct guess
    }
    network->reset_values();
    i++;
    network->cur_img = i;
  }
  std::cout << "Accuracy: " << correct << "/" << network->test_limit
            << " = " << correct / float(network->test_limit) << '\n';

  return 0;
}
