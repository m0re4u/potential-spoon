/**
 * Main function for running an SNN on the MNIST dataset
 *
 * Author: Michiel van der Meer
 */
#include <iostream>
#include <typeinfo>
#include <random>

#include "mnist/mnist_reader.hpp"
#include "CImg/CImg.h"

using namespace cimg_library;

/**
* Show the image at index in the training/test set
*/
void show_image(std::vector<unsigned char, std::allocator<unsigned char>> &vec) {
  CImg<uint8_t> img(28,28,1,1);
  unsigned pixel_ = 0;
  cimg_forXY(img,x,y) {  // Do 2 nested loops
    pixel_ = y * 28 + x;
    // std::cout << (unsigned) y  << '\n';
    // std::cout << (unsigned) x % 28  << '\n';
    img(x,y) = vec[pixel_];
  }

  img.display("First image");
}

int main(int argc, char const *argv[]) {
  std::cout << "Reading in MNIST dataset.." << '\n';

  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
  std::cout << "Training set size:  " << dataset.training_images.size() << '\n';
  std::cout << "Test set size:      " << dataset.test_images.size() << '\n';
  std::cout << "Image size: " << dataset.training_images[0].size() << '\n';

  // Test to see if images have loaded
  // show_image(dataset.training_images[100]);

  constexpr unsigned TIME = 350;
  constexpr unsigned DT = 1;
  int intensity = 100;
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dist(0, 1);

  // Generating spike train from pixel value
  double firing_rate = intensity / 4;
  std::cout << "Firing rate: " << firing_rate << '\n';
  std::cout << "Simulation time: " << TIME << '\n';
  std::cout << "Simulation step: " << DT << '\n';

  int spikes = 0;
  for (size_t i = 0; i < TIME; i += DT) {
    double num = dist(gen);
    if (num <= firing_rate * (DT / 1000.)) {
      std::cout << "Spike at " << i << '\n';
      spikes++;
    }
  }

  std::cout << "# of spikes: " << spikes << '\n';
  return 0;
}
