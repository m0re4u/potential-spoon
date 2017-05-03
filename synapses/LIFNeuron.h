/**
 * Models a LIF neuron
 * Author: Michiel van der Meer
 */
struct neuron {
   neuron() : threshold(1), refractory(0), V(0), tau(10), incomingSpike(0) {}
   double threshold; // threshold
   double refractory; // refractory period counter
   double V; // current potential
   double I; // current impulse
   int tau; // timing constant
   std::vector<unsigned> spikes;
   double incomingSpike; // incoming spike from the last cycle
};

/**
 * Calculates the updates for a neuron's potential
 * @param  I    Impulse current - will most likely be 0
 * @param  V    Current voltage of a neuron
 * @param  tau  timing constant
 * @return the change in potential
 */
double deltaV(double I, double V, int tau);
