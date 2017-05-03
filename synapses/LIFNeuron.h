/**
 * Models a LIF neuron
 * Author: Michiel van der Meer
 */
struct neuron {
  neuron() : threshold(1), refractory(0), V(0), I(0), tau(100), a_pre(0), a_post(0) {}
  double threshold; // threshold
  double refractory; // refractory period counter
  double V; // current potential
  double I; // current impulse
  int tau; // timing constant
  // std::vector<unsigned> spikes;
  std::vector<std::tuple<int, double>> incoming_spikes; // incoming spikes from the last cycle

  double a_pre; // trace of pre synaptic activity
  double a_post; // trace of post synaptic activity
};

/**
 * Calculates the updates for a neuron's potential
 * @param  I    Impulse current - will most likely be 0
 * @param  V    Current voltage of a neuron
 * @param  tau  timing constant
 * @return the change in potential
 */
double deltaV(double I, double V, int tau);
