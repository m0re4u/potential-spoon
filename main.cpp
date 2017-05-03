/**
 * Main function for running an SNN on the MNIST dataset
 *
 * Author: Michiel van der Meer
 */
#include <iostream>

#include "mnist/mnist_reader.hpp"

int main(int argc, char const *argv[]) {
  std::cout << "Reading in MNIST dataset.." << '\n';

  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  std::cout << "Test set size:      " << dataset.test_images.size() << '\n';


  return 0;
}
