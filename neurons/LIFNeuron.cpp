/**
 * Implementation of a Leaky-Integrate-and-Fire neuron
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <tuple>

#include "LIFNeuron.h"

#define N_NEURONS 6

double deltaV(double I, double V, int tau) {
  return (I - V) / tau;
}

double clip(double low, double high, double value) {
  if (value > high) {
    return high;
  } else if (value < low) {
    return low;
  } else {
    return value;
  }
}

int main(int argc, char const *argv[]) {
  int print_neuron = 0;
  if (argc == 2) {
    print_neuron = atoi(argv[1]);
  }
  // setup parameters and state variables
  int N = N_NEURONS; // number of neurons
  int T = 1000; // total time to simulate
  int ms = 1; // simulation time step
  double taupre = 20*ms;
  double taupost = 20*ms;
  double A_pre = 0.01; // update set for STDP for presynaptics
  double A_post = -A_pre * taupre / taupost * 1.05; // update set for STDP for prostsynaptics
  double w_max = 2; // maximum weight

  neuron neurons[N_NEURONS];

  // Set neuron specific parameters for input neuron
  neurons[0].tau = 10;
  neurons[0].I = 2;

  // Matrix indicating the connection weights between neurons
  double connectionMatrix[N_NEURONS][N_NEURONS] =
    { {0,1,0,0,0,0},
      {0,0,1,0,0,0},
      {0,0,0,1,0,0},
      {0,0,0,0,1,0},
      {0,0,0,0,0,1},
      {0,0,0,0,0,0} };

  // Simulate time ticks
  for (size_t t = 0; t < T; t += ms) {
    // Simulate each neuron
    for (size_t n = 0; n < N; n++) {
      // Handle presynaptics spikes
      if (neurons[n].incoming_spikes.size() >= 1) {
        for (auto& pre_spiked : neurons[n].incoming_spikes) {
          neurons[n].a_pre += A_pre;
          double new_w = connectionMatrix[std::get<0>(pre_spiked)][n] += neurons[n].a_post;
          connectionMatrix[std::get<0>(pre_spiked)][n] = clip(0, w_max, new_w);
          neurons[n].V += std::get<1>(pre_spiked);
        }
      }
      neurons[n].incoming_spikes.clear();

      if (argc > 1 && n == print_neuron) {
        std::cout << t << ", " << neurons[n].V << '\n';
      }

      // Check if the neuron fires
      if (neurons[n].V > neurons[n].threshold) {
        // std::cout << "Neuron " << n << " spiked at " << t << " with value " << neurons[n].V << '\n';
        // Set refractory period
        neurons[n].refractory = 3*ms;
        // Postsynaptic spike occurs
        neurons[n].a_post += A_post;
        // Spike to connected neurons
        for (size_t p = 0; p < N; p++) {
          if (connectionMatrix[n][p] != 0) {
            // std::cout << "Spiking to: " << p << " with weight "
            //           << connectionMatrix[n][p] << '\n';
            neurons[p].incoming_spikes.push_back(std::make_tuple(n, connectionMatrix[n][p]));
            // Update weight for synapse
            connectionMatrix[n][p] += neurons[n].a_pre;
          }
        }
        neurons[n].V = 0;
      } else if (neurons[n].refractory > 0) {
        // Still recovering from firing
        neurons[n].refractory -= ms;
        neurons[n].V = 0;
      } else {
        // Update potential
        double dv = deltaV(neurons[n].I, neurons[n].V, neurons[n].tau);
        neurons[n].V += dv;
      }
    }
  }

  return 0;
}
