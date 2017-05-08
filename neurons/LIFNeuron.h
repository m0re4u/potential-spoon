#pragma once
/**
 * Models a Leaky Integrate and Fire neuron
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include <iostream>
#include <vector>
#include <tuple>

struct neuron {
  neuron() {};
  unsigned index = 0;
  double threshold = 1; // threshold
  double refractory = 0; // refractory period counter
  double V = 0; // current potential
  double I = 0; // current impulse
  int tau = 100; // timing constant
  // std::vector<unsigned> spikes;
  std::vector<std::tuple<int, double>> incoming_spikes; // incoming spikes from the last cycle

  double a_pre = 0; // trace of pre synaptic activity
  double a_post = 0; // trace of post synaptic activity
};

/**
 * Calculates the updates for a neuron's potential
 * @param  I    Impulse current - will most likely be 0
 * @param  V    Current voltage of a neuron
 * @param  tau  timing constant
 * @return the change in potential
 */
double deltaV(double I, double V, int tau);

/**
 * Clip a value between two given numbers s.t. low < value < high
 * @param  low
 * @param  high
 * @param  value
 * @return
 */
double clip(double low, double high, double value);
