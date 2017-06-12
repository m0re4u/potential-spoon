#pragma once
/**
 * Models a network of LIF neurons using a computationally different model
 * Uses work from Izhikevich(https://www.izhikevich.org/publications/spnet.cpp)
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */


// Viewing images
#include "CImg/CImg.h"
// Reading in json configuration
#include "json/json.hpp"

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <cassert>
#include <iomanip>

using json = nlohmann::json;

class COMPlifnetwork {
public:
  COMPlifnetwork() = default;

  // Overall simulation variables
  unsigned stime_ = 0;          // current second of the simulation
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
  static constexpr int D = 20;       // maximal axonal conduction delay

  // Parameters of the network
  // Ordering in arrays of neurons: [Ne,Ni,Nd]
  float sm = 10.0;           // maximal synaptic strength
  std::vector<int> post[N];  // indices of postsynaptic neurons(connection M FROM N goes to post[N][M])
  std::vector<std::vector<float>*> s, sd;   // matrix of synaptic weights and their derivatives
  short delays_length[N][D];        // distribution of delays
  std::vector<short> delays[N][D];  // List of connections from neuron N having delay D
  std::vector<int> D_pre[N]; //
  int N_pre[N];              // Number of presynaptic connectionsk
  std::vector<int> I_pre[N]; // presynaptic information
  // Might cause issues, should be a pointe
  std::vector<std::vector<float>*> s_pre;  // presynaptic connection weights
  std::vector<std::vector<float>*> sd_pre; // presynaptic connection weights derivatives
  float LTP[N][501+D], LTD[N]; // STDP functions
  float a[N], d[N]; // neuronal dynamics parameters
  float v[N], u[N]; // voltage, recovery variables
  unsigned N_firings;    // the number of fired neurons
  static constexpr int N_firings_max=10000*N; // upper limit on the number of fired neurons per sec
  std::vector<std::tuple<int,int>> firings; // timing and index of spikes

  int highestSpikes[N][2];

  // Random generators for spike generation(Poisson distribution)
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dist1;
  std::uniform_real_distribution<> dist_weights;

  // Dataset used as input
  std::vector<std::vector<unsigned char, std::allocator<unsigned char>>> data;
  std::vector<unsigned char> labels;
  json config;

  /**
   * Initializes the paramters for the network. Derived from the Izhikevich
   * implementation(see this file's header)
   */
  void initialize_params(json config);

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
  bool generate_spike(unsigned pixel);

  /**
   * Handle the first layer of the network
   * @param image_index current image being presented to the network
   */
  void input_spikes();

  /**
   * Check whether input should be presented(350ms) and provide input, or
   * let the input layer sleep(150ms)
   */
  void presentData();

  /**
   * Run one cycle(500ms) of the network, presenting one image
   * @param learning whether adjusting the weights of the synapse should be turned on
   */
  void cycle(bool learning);

  /**
   * Prepare the network for the next cycle
   * @param learning whether adjusting the weights of the synapse should be turned on
   */
  void prepare(bool learning);

  /**
   * Handle a spike based on the neuron index
   * @param index
   * @param learning whether adjusting the weights of the synapse should be turned on
   */
  void handleSpikes(int index, bool learning);

  /**
   * Process spikes that have a delay
   * @param k
   * @param inputCurrent
   * @param learning whether adjusting the weights of the synapse should be turned on
   */
  void processDelayedSpikes(int k, float inputCurrents[], bool learning);

  /**
   * Update the voltages for a given neuron
   * @param index
   * @param inputCurrent
   */
  void updatePotential(int index, float inputCurrent);

  /**
   * label the neurons with the class it presented the highest response on
   */
  void labelNeurons();

  /**
   * Based on the presentation of one test image, get the class
   * @return the label predicted by the network
   */
  int getLabelFromSpikes();
};
