/**
 * Main function for running an SNN on the MNIST dataset
 *
 * Author: Michiel van der Meer
 */
#include <iostream>
#include <typeinfo>

#include "mnist/mnist_reader.hpp"
#include "CImg/CImg.h"

using namespace cimg_library;

int main(int argc, char const *argv[]) {
  std::cout << "Reading in MNIST dataset.." << '\n';

  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  std::cout << "Test set size:      " << dataset.test_images.size() << '\n';
  std::cout << "Image size: " << dataset.training_images[0].size() << '\n';

  CImg<uint8_t> img(28,28,1,1);
  unsigned index = 0;
  cimg_forXY(img,x,y) {  // Do 2 nested loops
    index = y * 28 + x;
    std::cout << index << '\n';
    // std::cout << (unsigned) y  << '\n';
    // std::cout << (unsigned) x % 28  << '\n';
    img(x,y) = dataset.training_images[600][index];
  }

  img.display("First image");

  return 0;
}
