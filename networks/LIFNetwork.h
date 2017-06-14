#pragma once
/**
 * Models a network of LIF neurons
 * Uses work from Izhikevich(https://www.izhikevich.org/publications/spnet.cpp)
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "Eigen/Dense"

#include <iostream>
#include "CImg/CImg.h"

#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <cassert>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <ctime>

class LIFNetwork {
public:
  LIFNetwork() = default;

  // Overall simulation variables
  unsigned mstime_ = 0;         // current millisecond of the simulation
  unsigned cycle_switcher = 0;  // counter between sleeping input/active input
  unsigned cur_img = 0;         // current image being presented
  unsigned input_spikes = 0;    // number of spikes in the input layer during the presentation of the current image
  unsigned image_spikes = 0;    // number of spikes in the exc layer during the presentation of the current image
  unsigned input_intensity = 0; // input intensity of the current image
  bool sleepingCycle = false;   // whether the input is active or sleeping
  bool learning = true;         // whether the connection weights are being adjusted using STDP
  bool plotting = false;        // output all neuron states per cycle for plotting
  bool record_training = false; // record the spikes during training

  // Constants used for the simulation
  unsigned SLEEP_TIME = 100; // no. of sleeping cycles
  unsigned IMG_TIME = 350; // no. of active input cycles
  unsigned BOTH_TIME = SLEEP_TIME + IMG_TIME;

  // Constants used in the neuron network setup
  static constexpr int Ne = 400;     // excitatory neurons
  static constexpr int Ni = Ne;      // inhibitory neurons
  static constexpr int Nn = Ne+Ni;   // all non-input neurons
  static constexpr int Nd = 784;     // input neurons
  static constexpr int N = Ne+Ni+Nd; // total number of neurons
  static constexpr int max_delay = 40;
  double ms = 0.001;
  double dt = 0.1*ms;
  double t = 1 * dt;
  double taue = 0.01;
  double tau_trace_pre = 0.00005;
  double trace_plus = 1;
  float theta_plus = 0.001;
  float tau_theta = 0.00001;

  int train_limit = 100; // number of images processed in the training stage
  int label_limit = 100; // number of images processed in the labelling stage
  int test_limit = 200;  // number of images processed in the testing stage

  static constexpr double v_reset_e = 0.;
  static constexpr double v_reset_i = 0.;
  static constexpr double v_thresh_e = 0.013;
  static constexpr double v_thresh_i = 0.025;
  // static constexpr double stdp_lr_pre = 0.0000001;
  static constexpr double stdp_lr_pre = 0.00001;
  static constexpr double stdp_offset = 0.1;
  static constexpr double wmax = 0.0009;
  static constexpr double wmin = 0;

  Eigen::Matrix<double, 1, N> S;

  std::vector<std::vector<int>*>   connectionTargets;
  std::vector<std::vector<int>*>   connectionDelays;
  std::vector<std::vector<float>*> connectionWeights;
  std::vector<float> connectionTrace;
  std::vector<float> thetas;

  cimg_library::CImg<unsigned char> im;
  cimg_library::CImgDisplay dis;

  std::vector<std::tuple<int, int, int>> firings;
  int refractory[N];
  int neuronClass[N];
  float previousSpike[N]; // store the timestamp of the previous spike
  double spikeQueue[max_delay][Ne];

  // Dataset used as input
  std::vector<std::vector<unsigned char, std::allocator<unsigned char>>> data;
  std::vector<unsigned char> labels;

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
   * output voltage for all neurons per cycle to cerr
   */
  void plotNeurons();

  void plotWeights();

  void plotWeightImage();

  void plotTrace();

  void plotFiringRates();

  /**
   * Output the current weight values to a file
   */
  void saveWeights();
  void saveStates();

  void showWeightExtrema();
  void showThetaExtrema();
  void showNeuronStates();
  void showTraces();

  void liveWeightUpdates();

};
