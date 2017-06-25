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

void trainLIF(Network* network, bool show, bool save) {
  // Run simulation
  bool showWeight = true;
  int shown = 0;
  int im = 0;
  int last_img = 0;
  while(im < network->train_limit) {
    if (last_img != network->cur_img) {
      im++;
      last_img = network->cur_img;
    }
    network->cycle();
    network->t += network->dt;
    network->mstime_++;
    if (showWeight) {
      if (show) {
        network->liveWeightUpdates();
      }
      if (save) {
        std::cout << std::to_string(im) << '\n';
        network->saveWeights("../weights/weights"+std::to_string(im)+".csv");
        network->saveThetas("../weights/thetas"+std::to_string(im)+".csv");
      }
      showWeight = false;
      shown = im;
    }
    if (im % 1000 == 0 && shown != im) {
      showWeight = true;
    }
    // std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
    //           << (network->cur_img / float(network->train_limit))<< std::flush;

  }
  std::cout << '\n';

  std::cout << "Outputting training statistics" << '\n';
  // network->plotSpikes();
  // network->plotWeights();
  // network->plotFiringRates();
  // network->plotWeightImage();
  if (save) {
    network->saveWeights("../weights/weights"+std::to_string(im)+".csv");
    network->saveThetas("../weights/thetas"+std::to_string(im)+".csv");
  }
  network->showWeightExtrema();
  network->showThetaExtrema();

  if (show) {
    network->liveWeightUpdates();
  }

}
void labelLIF(Network* network, bool save) {
  std::cout << "Labelling neurons.." << '\n';
  network->labelNeurons();
  if (save) {
    network->saveNeuronClasses("../weights/classes.csv");
  }
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
                << "ms with firings: " << network->input_spikes + network->exc_spikes + network->inh_spikes
                << '\n';
      std::cerr << network->input_spikes + network->exc_spikes + network->inh_spikes << ", "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << ", "
                << network->lastIntensity << ", "
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
  // Save weight every x iterations
  bool save = false;
  // Perform training
  bool train = false;
  // Label data after training
  bool label = false;
  // Save labels given to exc neurons
  bool save_labels = false;
  // Evaluate data after training
  bool eval = true;
  // Output cycle timings
  bool timings = true;

  LIFNetwork* n1 = new LIFNetwork(1, 1, 9999, true, r_t);
  // Opt1Network* n1 = new Opt1Network(1, 1, 9999, true, r_t);

  std::cout << "Reading in MNIST dataset.." << '\n';
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  n1->load_dataset(dataset.training_images, dataset.training_labels);

  std::cout << "Initializing parameters" << '\n';
  n1->initialize_params();
  std::cout << "Finished initializing parameters" << '\n';
  n1->showWeightExtrema();
  n1->showThetaExtrema();

  auto begin = std::chrono::high_resolution_clock::now();

  if (s_w) {
    // width, height = sqrt(Ne) * 28
    n1->im = cimg_library::CImg<unsigned char>(560,560,1,1,0);
    n1->dis = cimg_library::CImgDisplay(n1->im);
  }

  if (train) {
    trainLIF(n1, s_w, save);
  } else {
    n1->loadWeights("../weights/weights.csv");
    n1->loadThetas("../weights/thetas.csv");
    n1->showWeightExtrema();
    n1->showThetaExtrema();
    if (s_w) {
      n1->liveWeightUpdates();
    }
  }

  std::cout << "Resetting values" << '\n';
  n1->learning = false;
  n1->reset_values();

  if (label) {
    labelLIF(n1, save_labels);
  } else {
    n1->loadNeuronClasses("../weights/classes.csv");
  }

  if (eval) {
    n1->load_dataset(dataset.test_images, dataset.test_labels);
    testLIF(n1, timings);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << "ms" << '\n';

  if (s_w) {
    n1->im.display();
  }

  return 0;
}
