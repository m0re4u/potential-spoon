#pragma once
/**
 * Models a network of LIF neurons
 * Uses work from Izhikevich(https://www.izhikevich.org/publications/spnet.cpp)
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */


// Viewing images
#include "CImg/CImg.h"
// Reading in json configuration
#include "json/json.hpp"
// X11 also defines Success, but I'm not using it so unset it
#undef Success
#include "Eigen/Dense"

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cassert>
#include <iomanip>
#include <math.h>

using json = nlohmann::json;

class LIFNetwork {
public:
  LIFNetwork() = default;

  // Overall simulation variables
  unsigned mstime_ = 0;         // current millisecond of the simulation
  unsigned cycle_switcher = 0;  // counter between sleeping input/active input
  unsigned cur_img = 0;         // current image being presented
  bool sleepingCycle = false;   // whether the input is active or sleeping

  // Constants used for the simulation
  const unsigned SLEEP_TIME = 150; // no. of sleeping cycles
  const unsigned IMG_TIME = 350; // no. of active input cycles
  const unsigned BOTH_TIME = SLEEP_TIME + IMG_TIME;

  // Constants used in the neuron network setup
  static constexpr int Ne = 400;     // excitatory neurons
  static constexpr int Ni = Ne;      // inhibitory neurons
  static constexpr int Nn = Ne+Ni;   // all non-input neurons
  static constexpr int Nd = 784;     // input neurons
  static constexpr int N = Nd+Ne+Ni; // total number of neurons
  // static constexpr int D = 20;       // maximal axonal conduction delay
  static constexpr double mV = 1e-3;
  static constexpr int max_delay = 100;
  double ms = 1e-3;
  double dt = 0.1*ms;
  double t = 0 * ms;
  double taum = 20*ms;
  double taue = 1*ms;
  double taui = 2*ms;
  double duration = 1000*ms;

  static constexpr double v_rest_e = -65*mV;
  static constexpr double v_rest_i = -60*mV;
  static constexpr double v_reset_e = -65*mV;
  static constexpr double v_reset_i = -45*mV;
  static constexpr double v_thresh_e = -52*mV;
  static constexpr double v_thresh_i = -40*mV;

  Eigen::Matrix<double, 3, 3> A;
  Eigen::Matrix<double, 3, N> S;

  std::vector<std::vector<int>*> connectionTargets;
  std::vector<std::vector<int>*> connectionDelays;
  std::vector<std::vector<float>*> connectionWeights;

  std::vector<std::tuple<int, int>> firings;
  std::vector<double> state;
  int refractory[N];
  int highestSpikes[N][2];
  double spikeQueue[max_delay][Ne];

  // Random generators for spike generation(Poisson distribution)
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dist1;
  std::uniform_real_distribution<> dist_weights;
  std::uniform_real_distribution<> dist_delay;

  // Dataset used as input
  std::vector<std::vector<unsigned char, std::allocator<unsigned char>>> data;
  std::vector<unsigned char> labels;
  json config;

  /**
   * Initializes the paramters for the network. Derived from the Izhikevich
   * implementation(see this file's header)
   */
  void initialize_params();

  /**
   * Load in the dataset given
   * @param dataset data to be loaded in
   */
  void load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels);

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
  bool generateSpike(unsigned pixel);

  /**
   * Handle the first layer of the network
   */
  void inputSpikes();

  /**
   * Check whether input should be presented(350ms) and provide input, or
   * let the input layer sleep(150ms)
   */
  void presentData();

  /**
   * Run one cycle(500ms) of the network, presenting one image
   */
  void cycle();

  /**
   * Handle a spike based on the neuron index
   * @param index
   * @param learning whether adjusting the weights of the synapse should be turned on
   */
  void handleSpikes(int index);

  /**
   * label the neurons with the class it presented the highest response on
   */
  void labelNeurons();

  /**
   * Based on the presentation of one test image, get the class
   * @return the label predicted by the network
   */
  int getLabelFromSpikes();

  /**
   * output spikes per cycle to cerr
   */
  void plotSpikes();
  /**
   * output voltage for a given neuron per cycle to cerr
   */
  void plotNeuron();

};
