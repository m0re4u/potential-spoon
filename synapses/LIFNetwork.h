#pragma once
/**
 * Models a network of LIF neurons
 * Uses work from Izhikevich(https://www.izhikevich.org/publications/spnet.cpp)
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
  unsigned stime_ = 0; // current second of the simulation
  unsigned mstime_ = 0; // current millisecond of the simulation
  unsigned cycle_switcher = 0; // counter between sleeping input/active input
  unsigned image_index = 0; // current image being presented
  bool sleepingCycle = false; // whether the current cycle is sleeping

  // Constants used for the simulation
  const double DT = 0.001; // time step for spike generation(in seconds)
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
  // static constexpr int M = 100;      // the number of synapses per neuron

  // Parameters of the network
  // Ordering in arrays of neurons: [Ne,Ni,Nd]
  float sm = 10.0;           // maximal synaptic strength
  std::vector<int> post[N];  // indices of postsynaptic neurons(connection M FROM N goes to post[N][M])
  std::vector<float> s[N], sd[N];   // matrix of synaptic weights and their derivatives
  short delays_length[N][D];        // distribution of delays
  std::vector<short> delays[N][D];  // List of connection from neuron N having delay D
  // int N_pre[N], I_pre[N][3*M], D_pre[N][3*M];  // presynaptic information
  int N_pre[N];              // Number of presynaptic connectionsk
  std::vector<int> I_pre[N]; // presynaptic information
  std::vector<float> *s_pre[N], *sd_pre[N]; // presynaptic connection weights
  float LTP[N][1001+D], LTD[N]; // STDP functions
  float a[Nn], d[Nn]; // neuronal dynamics parameters
  float v[N], u[N]; // activity variables
  int N_firings;    // the number of fired neurons
  static constexpr int N_firings_max=100*N; // upper limit on the number of fired neurons per sec
  int firings[N_firings_max][2]; // indices and timings of spikes

  // Random generators for spike generation(Poisson distribution)
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dist;

  // data
  std::vector<std::vector<unsigned char, std::allocator<unsigned char>>> data;
  // Check where the spikes occur
  std::vector<std::vector<unsigned>> img_spikes;
  // Input layer
  std::vector<neuron> input_layer;

  /**
   * Initializes the paramters for the network. Derived from the Izhikevich
   * implementation(see this file's header)
   */
  void initialize_params();

  /**
   * Load in the dataset given
   * @param dataset data to be loaded in
   */
  void load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset);

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
   * Run one cycle(one second) of the network
   */
  void cycle();

  /**
   * Prepare the network for the next cycle
   */
  void prepare();
};
