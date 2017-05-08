#pragma once
/**
 * Models a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "neurons/LIFNeuron.h"
#include <iostream>
#include <vector>
#include <random>

class LIFNetwork {
public:
  LIFNetwork();

  // Overall simulation variables
  unsigned stime_ = 0; // current timestamp of the simulation
  unsigned cycle_switcher = 0; // counter between sleeping input/active input
  unsigned image_index = 0; // current image being presented
  bool sleepingCycle = false; // whether the current cycle is sleeping

  // Constants
  const double DT = 0.001; // time step for
  const unsigned SLEEP_TIME = 150;
  const unsigned IMG_TIME = 350;
  const unsigned BOTH_TIME = SLEEP_TIME + IMG_TIME;

  // Random generators for spike generation(Poisson distribution)
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dist;

  // Input layer
  std::vector<neuron> input_layer;
  /**
   * Show an image from the loaded in MNIST dataset
   * @param vec vector containing image data
   */
  void show_image(std::vector<unsigned char, std::allocator<unsigned char>> &vec);

  /**
   * Check if the spike train returns a spike at the current cycle, dependent
   * on the intensity of the pixel
   * @param  pixel index for the current pixel
   * @return true if there is a spike
   */
  bool generate_spike(unsigned pixel);

  /**
   * Handle the first layer of the network
   * @param image_index current image being presented to the network
   */
  void input_spikes(unsigned image_index);

  /**
   * Run one cycle of the network
   */
  void cycle();
};
