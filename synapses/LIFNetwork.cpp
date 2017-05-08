/**
 * Models a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "LIFNetwork.h"
#include "CImg/CImg.h"


LIFNetwork::LIFNetwork() {
  std::mt19937 generator(this->rd());
  this->gen = generator;
  std::uniform_real_distribution<> distribution(0, 1);
  this->dist = distribution;
}

void LIFNetwork::show_image(std::vector<unsigned char, std::allocator<unsigned char>> &vec) {
  cimg_library::CImg<uint8_t> img(28,28,1,1);
  unsigned pixel_ = 0;
  cimg_forXY(img,x,y) {  // Do 2 nested loops
    pixel_ = y * 28 + x;
    img(x,y) = vec[pixel_];
  }

  img.display("First image");
}

bool LIFNetwork::generate_spike(unsigned pixel) {
  // Generating spike train from pixel value
  double firing_rate = 100 / 4;
  double num = this->dist(this->gen);
  return num <= firing_rate * this->DT;
}

void LIFNetwork::input_spikes(unsigned image_index) {
  std::cout << "Processing image: " << image_index << '\n';
  for (std::size_t i = 0, e = this->input_layer.size(); i != e; ++i) {
    bool spike = generate_spike(i);
    if (spike) {
      std::cout << "Spiking pixel at index " << i << '\n';
    }
  }
}

void LIFNetwork::cycle() {
  std::cout << "Cycle " << this->stime_ << '\n';
  if (this->sleepingCycle) {
    // Check state of the next cycle
    this->cycle_switcher++;
    if (this->cycle_switcher >= this->SLEEP_TIME) {
      this->sleepingCycle = false;
      this->cycle_switcher = 0;
    }
  } else {
    this->input_spikes(this->stime_ / this->BOTH_TIME);
    // Check state of the next cycle
    this->cycle_switcher++;
    if (this->cycle_switcher >= this->IMG_TIME) {
      this->cycle_switcher = 0;
      this->sleepingCycle = true;
    }
  }
  this->stime_++;
}
