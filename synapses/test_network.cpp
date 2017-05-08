 /**
 * Models a network of LIF neurons
 *
 * Author: Michiel van der Meer
 */

#include <iostream>
#include <algorithm>

int main(int argc, char *argv[])
{
  std::cout << "Running simple SNN.." << std::endl;

  // setup parameters for overall simulation
  const int T = 10;    // total time to simulate
  const int dt = 1;     // simulation time step
  const int neurons_n = 10; // number of neurons
  int t_rest[neurons_n]; // initial refractory time
  std::fill_n(t_rest, neurons_n, 0);

  // LIF properties per neuron
  int Rm[neurons_n]; // resistance (kOhm)
  int Cm[neurons_n]; // capacitance (uF)
  int tau_m[neurons_n]; // time constant (msec)
  int tau_ref[neurons_n]; // refractory period (msec)
  int Vth[neurons_n]; // spike threshold (V)
  double V_spike[neurons_n]; // spike delta (V)

  // Fill LIF properties for all neurons
  std::fill_n(Rm, neurons_n, 1);
  std::fill_n(Cm, neurons_n, 10);
  for (size_t j = 0; j < neurons_n; j++) {
    tau_m[j] = Rm[j] * Cm[j];
  }
  std::fill_n(tau_ref, neurons_n, 4);
  std::fill_n(Vth, neurons_n, 1);
  std::fill_n(V_spike, neurons_n, 0.5);

  // spike potential over time
  double Vm[T][neurons_n];
  for (size_t k = 0; k < T; k++) {
    std::fill_n(Vm[k], neurons_n, 0);
  }
  // Stimulus
  double I = 0;
  double weight = 1;

  // Connection matrix
  int weights[neurons_n][neurons_n];
  for ( int m = 0; m < neurons_n; m++) {
    if (m+1 < neurons_n) {
      weights[m][m+1] = 1;
    }
  }

  // Simulate
  for (size_t t = 0; t < T; t += dt) {
    for (size_t n = 0; n < neurons_n; n++) {
      // Caclulate incoming impulse
      if (n == 0) {
        I = 1.5;
      } else {
        for (size_t c = 0; c < neurons_n; c++) {
          if (weights[n][c] == 1) {
            std::cout << "Connection between " << n << " and " << c << '\n';
            std::cout << "Upping impulse with " << Vm[t-1][c] << " to " << I << '\n';
            I += weights[n][c] * Vm[t-1][c];
          }
        }
      }
      std::cout << "Neuron: " << n << " Incoming impusle: " << I << '\n';
      I = 0;
      // Update membrame potential
      // if (t > t_rest[n]) {
      //   double impulse = I * weight
      //   Vm[t][n] = Vm[t-1][n] + (-Vm[t-1][n] + I*Rm[n]) / tau_m[n] * dt;
      //   if (Vm[t][n] >= Vth[n]) {
      //     Vm[t][n] += V_spike[n];
      //     t_rest[n] = t + tau_ref[n];
      //   }
      // }
      // std::cout << "Neuron: " << n << " potential: " << Vm[t][n] << '\n';
    }
  }

}
