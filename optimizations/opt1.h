#pragma once
/**
 * Models a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */
#include <iostream>
#include "CImg/CImg.h"
#include "networks/Network.h"

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
#include <sstream>

class Opt1Network : public Network {
public:
  Opt1Network(int train, int label, int test, bool learn, bool record) : Network(train, label, test, learn, record) {}

  // Overall simulation variables
  unsigned cycle_switcher = 0;  // counter between sleeping input/active input
  unsigned input_intensity = 0; // input intensity of the current image
  bool sleepingCycle = false;   // whether the input is active or sleeping
  bool plotting = false;        // output all neuron states per cycle for plotting

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
  double taue = 0.01;
  double trace_plus = 1;
  double tau_trace_pre = 0.00005;
  float theta_plus = 0.002;
  float tau_theta = 0.0000001;

  static constexpr double v_reset_e = 0.;
  static constexpr double v_reset_i = 0.;
  static constexpr double v_thresh_e = 0.013;
  static constexpr double v_thresh_i = 0.025;
  // static constexpr double stdp_lr_pre = 0.0000001;
  static constexpr double stdp_lr_pre = 0.0003;
  static constexpr double stdp_offset = 0.1;
  static constexpr double wmax = 0.0009;
  static constexpr double wmin = 0;

  double S[N];

  // exc neuron variables
  float excWeights[Ne][1]; // outgoing connection weights
  int excTargets[Ne][1];   // outgoing connection targets
  float thetas[Ne];        // threshold addition

  // inh neuron variables
  float inhWeights[Ni][Ne-1];
  int inhTargets[Ni][Ne-1];

  // input neuron variables
  float inputWeights[Nd][Ne];
  int inputTargets[Nd][Ne];
  int inputDelays[Nd][Ne];
  float connectionTrace[Nd];

  float* incomingWeights[Ne][Nd];

  int refractory[N]; // nonzero if in refractory state
  int neuronClass[N];
  float previousSpike[N]; // store the timestamp of the previous spike
  double spikeQueue[max_delay][Ne];
  int firingsPerNeuron[N];

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
  void handleExcSpikes(int index);
  void handleInhSpikes(int index);
  void handleInputSpikes(int index);

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
  void saveWeights(std::string filename);
  void loadWeights(std::string filename);
  void saveThetas(std::string filename);
  void loadThetas(std::string filename);
  void saveNeuronClasses(std::string filename);
  void loadNeuronClasses(std::string filename);
  void getImageAvgIntensity();
  void saveStates();

  void showWeightExtrema();
  void showThetaExtrema();
  void showNeuronStates();
  void showTraces();

  void liveWeightUpdates();

};
