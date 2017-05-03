/**
 * Models LIF neurons
 *
 * Author: Michiel van der Meer
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>

double deltaV(double v0, double V, int tau) {
  return (v0 - V) / tau;
}
struct neuron {
  neuron() : index(-1), threshold(1), refractory(0), V(0), nSpikes(0) {}
  int index;
  double threshold; // threshold
  double refractory; // refractory period counter
  double V; // current potential
  double v0;
  unsigned nSpikes;
} neurons[100];


int main(int argc, char *argv[])
{

  // Random seed
  srand(time(NULL));
  // setup parameters and state variables
  int N = 100; // number of neurons

  int T = 1000; // total time to simulate
  int ms = 1; // simulation time step
  int tau = 10 * ms; // time constant in deltaV()
  int v0_max = 3.0; // maximum value for v0, used in setting v0

  for (size_t i = 0; i < N; i++) {
    neurons[i].index = i;
    neurons[i].V = ((double) rand() / (RAND_MAX));
    neurons[i].v0 = (double) i * v0_max / (N-1);
  }

  // Simulate the neurons
  for (size_t t = 0; t < T; t += ms) {
    for (size_t n = 0; n < N; n++) {
      // Update potential
      neurons[n].V += deltaV(neurons[n].v0, neurons[n].V, tau);
      // Check if the neuron fires
      if (neurons[n].V > neurons[n].threshold) {
        neurons[n].V = 0;
        // std::cout << t << ", " << neurons[n].index << '\n';
        // spikeTimings.push_back(t);
        neurons[n].nSpikes++;
        // neurons[n].refractory = 3*ms;
        neurons[n].refractory = 0;
      } else if (neurons[n].refractory > 0) {
        neurons[n].refractory--;
        neurons[n].V = 0;
      }
    }
  }

  for (size_t neur = 0; neur < N; neur++) {
    std::cout << neurons[neur].nSpikes << '\n';
    // std::cout << neurons[neur].v0 << ", " << neurons[neur].nSpikes << '\n';
  }

}
