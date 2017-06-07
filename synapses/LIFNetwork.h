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
#include <algorithm>
#include <random>
#include <cassert>
#include <iomanip>
#include <fstream>
#include <math.h>

using json = nlohmann::json;

class LIFNetwork {
public:
  LIFNetwork() = default;

  // Overall simulation variables
  unsigned mstime_ = 0;         // current millisecond of the simulation
  unsigned cycle_switcher = 0;  // counter between sleeping input/active input
  unsigned cur_img = 0;         // current image being presented
  unsigned image_spikes = 0;    // number of spikes in the exc layer during the presentation of the current image
  unsigned input_intensity = 0; // input intensity of the current image
  bool sleepingCycle = false;   // whether the input is active or sleeping
  bool learning = true;         // whether the connection weights are being adjusted using STDP
  bool plotting = false;        // output all neuron states per cycle for plotting

  // Constants used for the simulation
  const unsigned SLEEP_TIME = 300; // no. of sleeping cycles
  const unsigned IMG_TIME = 350; // no. of active input cycles
  const unsigned BOTH_TIME = SLEEP_TIME + IMG_TIME;

  // Constants used in the neuron network setup
  static constexpr int Ne = 400;     // excitatory neurons
  static constexpr int Ni = Ne;      // inhibitory neurons
  static constexpr int Nn = Ne+Ni;   // all non-input neurons
  static constexpr int Nd = 784;     // input neurons
  static constexpr int N = Ne+Ni+Nd; // total number of neurons
  static constexpr double mV = 0.001;
  static constexpr int max_delay = 100;
  double ms = 0.001;
  double dt = 0.1*ms;
  double t = 0 * ms;
  double taue = 0.005; // 50 cycles
  double taui = 0.002; // 20 cycles
  double tau_trace_pre = 0.001;
  double tau_trace_post = 0.002;
  float theta_plus = 0.03;
  float tau_theta = 500;

  int train_limit = 10000; // number of images processed in the training stage
  int label_limit = 10000; // number of images processed in the labelling stage
  int test_limit = 1;  // number of images processed in the testing stage

  static constexpr double v_rest_e = 0*mV;
  static constexpr double v_rest_i = 0*mV;
  static constexpr double v_reset_e = 0*mV;
  static constexpr double v_reset_i = 20*mV;
  static constexpr double v_thresh_e = 13*mV;
  static constexpr double v_thresh_i = 25*mV;
  static constexpr double stdp_lr = 0.00001;
  static constexpr double wmax = 1.0;
  static constexpr double wmin = 0;

  Eigen::Matrix<double, 1, N> S;

  std::vector<std::vector<int>*>   connectionTargets;
  std::vector<std::vector<int>*>   connectionDelays;
  std::vector<std::vector<float>*> connectionWeights;
  std::vector<float> connectionTrace;
  std::vector<float> postTrace;
  std::vector<float> thetas;

  std::vector<std::tuple<int, int, int>> firings;
  int refractory[N];
  int neuronClass[N];
  float previousSpike[N]; // store the timestamp of the previous spike
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
   * Reset the values used in the network without modifying the weights, such
   * that we can run labelling/evaluation cycles after.
   */
  void reset_values();

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
   * Apply spikes that are timed to occur at the current cycle(delayed spikes)
   * @param i current neuron index
   */
  void processPreviousSpikes(int i);

  /**
   * Impose exponential decay on all neurons
   */
  void decayNeurons();

  /**
   * Impose exponential decay on the connection weight trace
   */
  void decayTrace();

  void decayTheta();

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
   */
  void handleSpikes(int index);

  /**
   * Update the weight values for the connections that are coming in to the
   * neuron at index
   * @param index update weights for connections to index
   */
  void updateIncomingWeights(int index);

  void updateFromInput(int index);
  /**
   * Update the weight values for the connectinons going out of the index
   * @param index
   */
  // void updateOutgoingWeights(int index);

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
  void plotNeurons();

  void plotWeights();

  void plotTrace();

  /**
   * Output the current weight values to a file
   */
  void saveWeights();
  void saveStates();

  void showWeightExtrema();
  void showThetaExtrema();

};
