/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>
// time
#include <chrono>

// Utilities
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// Network
#include "synapses/LIFNetwork.h"

#include <omp.h>

int main(int argc, char const *argv[]) {

  bool eval = false;
  bool r_t = true;


  LIFNetwork *network = new LIFNetwork();

  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  network->load_dataset(dataset.training_images, dataset.training_labels);

  std::cout << "Initializing parameters" << '\n';
  network->initialize_params();
  network->record_training = r_t;
  std::cout << "Finished initializing parameters" << '\n';

  network->showWeightExtrema();
  network->showThetaExtrema();

  // Run simulation
  while(network->cur_img < network->train_limit) {
    network->cycle();
    network->t += network->dt;
    network->mstime_++;
    network->plotNeurons();
    std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
              << (network->cur_img / float(network->train_limit))<< std::flush;
  }
  std::cout << '\n';

  std::cout << "Outputting training statistics" << '\n';
  // network->plotSpikes();
  // network->plotWeights();
  // network->plotFiringRates();
  network->saveWeights();
  network->showWeightExtrema();
  network->showThetaExtrema();

  // std::cout << "Outputting weight image data" << '\n';
  // network->plotWeightImage();

  if (!eval) {
    return 0;
  }

  std::cout << "Resetting values" << '\n';
  network->learning = false;
  network->reset_values();

  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();

  std::cout << "Evaluating test set" << '\n';
  network->load_dataset(dataset.test_images, dataset.test_labels);
  float correct = 0.;

  // Per image, predict a label and check if it is correct
  int i = 0;
  int responses[10] = {0};
  while (network->cur_img < network->test_limit) {
    network->showWeightExtrema();
    int label = network->getLabelFromSpikes();

    std::cout << "Index: " << i << " Guessed: " << label
              << " versus actual: " << int(network->labels[i]) << '\n';
    if (label == int(network->labels[i])) {
      correct++; // correct guess
    }
    responses[label]++;
    network->reset_values();
    i++;
    network->cur_img = i;
  }
  std::cout << "Accuracy: " << correct << "/" << network->test_limit
            << " = " << correct / float(network->test_limit) << '\n';

  std::cout << "Guesses: ";
  for (size_t i = 0; i < 10; i++) {
    std::cout << responses[i] << ", ";
  }
  std::cout << '\n';

  return 0;
}
