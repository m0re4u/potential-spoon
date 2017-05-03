/**
 * Models three LIF neurons an a synaptic connection
 *
 * Author: Michiel van der Meer
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>

double deltaV(double I, double V, int tau) {
  return (I - V) / tau;
}

struct neuron {
  neuron() : threshold(1), refractory(0), V(0), tau(10), incomingSpike(0) {}
  double threshold; // threshold
  double refractory; // refractory period counter
  double V; // current potential
  double I; // current impulse
  int tau; // timing constant
  std::vector<unsigned> spikes;
  double incomingSpike; // incoming spike from the last cycle
} neurons[3];



int main(int argc, char *argv[])
{

  // setup parameters and state variables
  int N = 3; // number of neurons
  int T = 100; // total time to simulate
  int ms = 1; // simulation time step

  // Set neuron specific parameters
  neurons[0].tau = 10;
  neurons[1].tau = 100;
  neurons[2].tau = 100;
  neurons[0].I = 2;
  neurons[1].I = 0;
  neurons[2].I = 0;

  // Matrix indicating the connection weights between neurons
  double connectionMatrix[3][3] = {{0,0.2,0.4},{0,0,0},{0,0,0}};

  // Simulate time ticks
  for (size_t t = 0; t < T; t += ms) {
    // Simulate each neuron
    for (size_t n = 0; n < N; n++) {
      // Assign possible incoming spikes
      neurons[n].V += neurons[n].incomingSpike;
      neurons[n].incomingSpike = 0;

      // Check if the neuron fires
      if (neurons[n].V > neurons[n].threshold) {
        std::cout << "Neuron " << n << " spiked at " << t << " with value " << neurons[n].V << '\n';
        neurons[n].spikes.push_back(t);
        neurons[n].refractory = 3*ms;

        // Spike to connected neurons
        for (size_t p = 0; p < N; p++) {
          if (connectionMatrix[n][p] != 0) {
            neurons[p].incomingSpike += connectionMatrix[n][p];
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

}
