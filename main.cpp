/**
 * Main function for running an SNN on the MNIST dataset
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include <iostream>
// Time
#include <chrono>
// OpenMP functions & variables
#include <omp.h>

// Utilities
#include "mnist/mnist_reader.hpp"

#include "networks/Network.h"
// Basic network
#include "networks/LIFNetwork.h"
// Optimized network
#include "optimizations/opt1.h"

void trainLIF(Network* network, bool show) {
  if (show) {
    network->im = cimg_library::CImg<unsigned char>(560,560,1,1,0);
    network->dis = cimg_library::CImgDisplay(network->im);
  }
  // Run simulation
  bool showWeight = true;
  int shown = 0;
  while(network->cur_img < network->train_limit) {
    network->cycle();
    network->t += network->dt;
    network->mstime_++;
    if (showWeight && show) {
      network->liveWeightUpdates();
      showWeight = false;
      shown = network->cur_img;
    }
    if (network->cur_img % 50 == 0 && shown != network->cur_img) {
      showWeight = true;
    }
    std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
              << (network->cur_img / float(network->train_limit))<< std::flush;

  }
  std::cout << '\n';

  std::cout << "Outputting training statistics" << '\n';
  // network->plotSpikes();
  // network->plotWeights();
  // network->plotFiringRates();
  // network->plotWeightImage();
  network->saveWeights("weights.csv");
  network->showWeightExtrema();
  network->showThetaExtrema();

  if (show) {
    network->liveWeightUpdates();
  }

}
void labelLIF(Network* network) {
  std::cout << "Resetting values" << '\n';
  network->learning = false;
  network->reset_values();

  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();
}

void testLIF(Network* network, bool timing) {
  std::cout << "Evaluating test set" << '\n';
  float correct = 0.;

  // Per image, predict a label and check if it is correct
  int i = 0;
  int responses[10] = {0};
  while (network->cur_img < network->test_limit) {
    auto begin = std::chrono::high_resolution_clock::now();
    int label = network->getLabelFromSpikes();
    std::cout << "Index: " << i << " Guessed: " << label
              << " versus actual: " << int(network->labels[i]);
    if (timing) {
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << " - Obtaining answer took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
      << "ms with firings: " << network->firings.size()
      << '\n';
      std::cerr << network->firings.size() << ", "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
      << '\n';
    } else {
      std::cout << '\n';
    }
    if (label == int(network->labels[i])) {
      correct++; // correct guess
    }
    responses[label]++;
    network->reset_values();
    i++;
    network->cur_img = i;
  }
  std::cout << "==== Results ====" << '\n';
  network->showWeightExtrema();
  network->showThetaExtrema();

  std::cout << "Accuracy: " << correct << "/" << network->test_limit
            << " = " << correct / float(network->test_limit) << '\n';

  std::cout << "Guesses: ";
  for (size_t i = 0; i < 10; i++) {
    std::cout << responses[i] << ", ";
  }
  std::cout << '\n';
}


int main(int argc, char const *argv[]) {

  std::cout << "OpenMP version: " << _OPENMP  << " - max threads: " << omp_get_max_threads() << '\n';
  // Record training spikes
  bool r_t = false;
  // Show weight progression
  bool s_w = false;
  // Perform training
  bool train = true;
  // Label data after training
  bool label = true;
  // Evaluate data after training
  bool eval = true;
  // Output cycle timings
  bool timings = true;

  // LIFNetwork* n1 = new LIFNetwork(5000, 5000, 500, true, r_t);
  Opt1Network* n1 = new Opt1Network(5000, 5000, 500, true, r_t);

  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  n1->load_dataset(dataset.training_images, dataset.training_labels);

  std::cout << "Initializing parameters" << '\n';
  network->initialize_params();
  std::cout << "Finished initializing parameters" << '\n';
  network->showWeightExtrema();
  network->showThetaExtrema();

  auto begin = std::chrono::high_resolution_clock::now();

  if (train) {
    trainLIF(n1, s_w);
  } else {
    network->loadWeights("weights.csv");
  }

  if (label || eval) {
    labelLIF(n1);
  }

  if (eval) {
    n1->load_dataset(dataset.test_images, dataset.test_labels);
    testLIF(n1, timings);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << "ms" << '\n';

  n1->im.display();

  return 0;
}
