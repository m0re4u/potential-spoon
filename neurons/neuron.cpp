/**
 * Models one LIF neuron
 *
 * Author: Michiel van der Meer
 */

#include <iostream>

int main(int argc, char *argv[])
{
  std::cout << "Running LIF model.." << std::endl;

  // setup parameters and state variables
  int T = 100;    // total time to simulate
  int dt = 1;     // simulation time step
  int t_rest = 0; // initial refractory time

  // LIF properties
  int Rm = 1;           // resistance (kOhm)
  int Cm = 10;          // capacitance (uF)
  int tau_m = Rm*Cm;    // time constant (msec)
  int tau_ref = 4;      // refractory period (msec)
  int Vth = 1;          // spike threshold (V)
  double V_spike = 0.5; // spike delta (V)

  // spike potential over time
  double Vm[T];
  for (size_t i = 0; i < T; i++) {
    Vm[i] = 0;
  }
  // Stimulus
  double I = 1.5;

  // Simulate!
  for (size_t t = 0; t < T; t += dt) {
    if (t > t_rest) {
      Vm[t] = Vm[t-1] + (-Vm[t-1] + I*Rm) / tau_m * dt;
      if (Vm[t] >= Vth) {
        Vm[t] += V_spike;
        t_rest = t + tau_ref;
      }
    }
    std::cout << "Neuron potential: " << Vm[t] << '\n';
  }


}
