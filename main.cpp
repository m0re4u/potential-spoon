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

// Basic network
#include "networks/LIFNetwork.h"
// Optimized network
#include "optimizations/opt1.h"

void trainLIF(LIFNetwork* network, int images, bool record, bool show) {
  std::cout << "Initializing parameters" << '\n';
  network->initialize_params();
  network->record_training = record;
  network->train_limit = images;

  if (show) {
    network->im = cimg_library::CImg<unsigned char>(560,560,1,1,0);
    network->dis = cimg_library::CImgDisplay(network->im );
  }
  std::cout << "Finished initializing parameters" << '\n';

  network->showWeightExtrema();
  network->showThetaExtrema();

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
    if (network->cur_img % 10 == 0 && shown != network->cur_img) {
      showWeight = true;
    }
    std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
              << (network->cur_img / float(network->train_limit))<< std::flush;

  }
  std::cout << '\n';

  std::cout << "Outputting training statistics" << '\n';
  network->plotSpikes();
  // network->plotWeights();
  // network->plotFiringRates();
  if (show) {
    network->liveWeightUpdates();
  }
  // network->plotWeightImage();

  network->showWeightExtrema();
  network->showThetaExtrema();

}
void labelLIF(LIFNetwork* network, int labeling) {
  std::cout << "Resetting values" << '\n';
  network->learning = false;
  network->reset_values();
  network->label_limit = labeling;

  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();
}

void testLIF(LIFNetwork* network, int testing) {
  std::cout << "Evaluating test set" << '\n';
  network->test_limit = testing;
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
}


int main(int argc, char const *argv[]) {

  // record training spikes
  bool r_t = true;
  // show weight progression
  bool s_w = true;
  // Label data after training
  bool label = true;
  // Evaluate data after training
  bool eval = true;

  LIFNetwork*   n = new LIFNetwork();
  Opt1Network* o1 = new Opt1Network();

  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

  n->load_dataset(dataset.training_images, dataset.training_labels);
  trainLIF(n, 50, r_t, s_w);

  if (label || eval) {
    labelLIF(n, 500);
  }
  if (eval) {
    n->load_dataset(dataset.test_images, dataset.test_labels);
    testLIF(n, 1);
  }


  return 0;
}
