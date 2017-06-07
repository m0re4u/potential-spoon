/**
 * Implements a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "LIFNetwork.h"
#include "mnist/mnist_reader.hpp"


void LIFNetwork::initialize_params() {
  // random number generation
  std::mt19937 generator(this->rd());
  this->gen = generator;
  std::uniform_real_distribution<> distribution1(0, 1);
  this->dist1 = distribution1;
  std::uniform_real_distribution<> distribution_weights(0, 0.3);
  this->dist_weights = distribution_weights;
  std::uniform_real_distribution<> distribution_delay(0, 100);
  this->dist_delay = distribution_delay;

  // Indexing variables
  int i, j;

  // state matrix
  S = Eigen::MatrixXd::Constant(1, N, 0);

  // Reset the values used as states, spike queue, and refractory counters
  reset_values();

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    auto targets = new std::vector<int>();
    auto delays = new std::vector<int>();
    auto weights = new std::vector<float>();
    if (i < Ne) {
      // exc neuron connection to inhibitory: one on one
      targets->push_back(i+Ne);
      weights->push_back(26*mV);
      postTrace.push_back(0);
    } else if (i >= Ne && i < Nn) {
      // inh neuron connection to all exc except incoming
      for (j = 0; j < Ne; j++) {
        if (!(i - Ne == j)) {
          targets->push_back(j);
          weights->push_back(-17.4*mV);
        }
      }
    } else {
      // input connections, all to all input to exc
      for (j = 0; j < Ne; j++) {
        targets->push_back(j);
        int d = this->dist_delay(this->gen);
        delays->push_back(d);
        float val = this->dist_weights(this->gen);
        weights->push_back(val*5*mV);
      }
      connectionTrace.push_back(0);
    }
    connectionTargets.push_back(targets);
    connectionDelays.push_back(delays);
    connectionWeights.push_back(weights);
  }
  for (size_t i = 0; i < Ne; i++) {
    thetas.push_back(0.002);
  }
}

void LIFNetwork::reset_values() {
  // Starting voltages
  for (size_t i = 0; i < Ne; i++) {
    S(0,i) = 0. * mV;
    // Spike delays only occur in input to exc connections - reset them
    for (size_t j = 0; j < max_delay; j++) {
      spikeQueue[j][i] = 0; // very cache inefficient :/
    }
  }
  for (size_t i = Ne; i < Nn; i++) {
    S(0,i) = 0. * mV;
  }
  for (size_t i = Nn; i < N; i++) {
    S(0,i) = -60; // does not matter, rate based
  }

  for (size_t i = 0; i < N; i++) {
    previousSpike[i] = 0;
    refractory[i] = 0; // no neuron is has fired, so no refraction
  }

  mstime_ = 0;
  t = 0;
  cycle_switcher = 0;
  cur_img = 0;
  sleepingCycle = false;
  firings.clear();
}

void LIFNetwork::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) {
  // reset simulation variables
  this->data = dataset;
  this->labels = labels;
  this->reset_values();
}

void LIFNetwork::show_image(std::vector<unsigned char, std::allocator<unsigned char>> &vec) {
  cimg_library::CImg<uint8_t> img(28,28,1,1);
  unsigned pixel_ = 0;
  cimg_forXY(img,x,y) {  // Do 2 nested loops
    pixel_ = y * 28 + x;
    img(x,y) = vec[pixel_];
  }
  img.display("Test image");
}

bool LIFNetwork::generateSpike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4000.; // per millisecond
  if (input_intensity > 0) {
    firing_rate = ((0.06375 + (input_intensity*0.032)) * firing_rate) / 0.06375;
  }
  double num = this->dist1(this->gen);
  return num <= firing_rate;
}

void LIFNetwork::inputSpikes() {
  for (int i = Nn; i != N; ++i) {
    assert(i - Nn >= 0);
    bool spike = generateSpike(this->data[this->cur_img][i - Nn]);
    if (spike) {
      S(0, i) = 0.005;
    } else {
      S(0, i) = -1;
    }
  }
}

void LIFNetwork::presentData() {
  if (sleepingCycle) {
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= SLEEP_TIME) {
      cur_img++;
      cycle_switcher = 0;
      sleepingCycle = false;
      image_spikes = 0;
    }
  } else {
    inputSpikes();
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= IMG_TIME) {
      if (image_spikes < 5) {
        // not enough activation, present the image again with a higher intensity
        // std::cout << "Not enough spikes, repeating image" << '\n';
        cycle_switcher = 0;
        input_intensity++;
      } else {
        cycle_switcher = 0;
        sleepingCycle = true;
        input_intensity = 0;
        image_spikes = 0;
      }
    }
  }
}
void LIFNetwork::processPreviousSpikes(int i) {
  // Add up any delayed spikes
  if (spikeQueue[mstime_% max_delay][i] > 0) {
    S(0, i) += spikeQueue[mstime_ % max_delay][i];
    spikeQueue[mstime_ % max_delay][i] = 0;
  }
}

void LIFNetwork::handleSpikes(int i) {
  // Fixes a bug with too much activation - should probably be fixed through
  // parameters
  if (std::isinf(S(0, i))) {
    S(0,i) = 0;
  }

  if (i < Ne) {
    if (refractory[i] > 0) {
      refractory[i]--;
      return;
    }
    if (S(0, i) > (v_thresh_e + thetas[i])) {
      // Spike in excitatory neuron
      // std::cout << "Spike in exc: " << i << '\n';
      if (learning) {
        // postsynaptic spike in neuron i -> update weight of incoming connection
        updateIncomingWeights(i);
        // update theta for homeostasis
        thetas[i] += theta_plus;
      }
      // Propagate spike, is done only once, since exc has one connection
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(0, (*connectionTargets[i])[j]) += (*connectionWeights[i])[j];
      }
      S(0, i) = v_reset_e;  // reset potential
      refractory[i] = 50;   // set refractory period
      postTrace[i] += 1;    // update trace for STDP
      if (!learning) {
        // Store spike
        int c = int(this->labels[this->cur_img]);
        firings.push_back(std::make_tuple(mstime_, i, c));
      }

      image_spikes++;       // count this spike for activation
      previousSpike[i] = t; // set timestamp as latest activation
    }
  } else if (i >= Ne && i < Nn) {
    if (refractory[i] > 0) {
      refractory[i]--;
      return;
    }
    if (S(0, i) > v_thresh_i) {
      // std::cout << "Spike in inh: " << i << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(0, (*connectionTargets[i])[j]) += (*connectionWeights[i])[j];
      }
      refractory[i] = 20;
      S(0, i) = v_reset_i;  // reset potential
      if (!learning) {
        // Store spike
        int c = int(this->labels[this->cur_img]);
        firings.push_back(std::make_tuple(mstime_, i, c));
      }
      previousSpike[i] = t;
    }
  } else {
    // Since the input is rate based, use an arbitrary threshold
    if (S(0, i) > 0) {
      // std::cout << "Spike in input: " << i << '\n';
      if (learning) {
        // check if this spikes right after a exc spike, and weaken the connection if it did
        updateFromInput(i);
      }
      connectionTrace[i-Nn] += 1;
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        // presynaptic spike
        spikeQueue[(mstime_ + (*connectionDelays[i])[j]) % max_delay][(*connectionTargets[i])[j]] += (*connectionWeights[i])[j];
      }
      S(0, i) = -1;  // reset potential
      if (!learning) {
        // Store spike
        int c = int(this->labels[this->cur_img]);
        firings.push_back(std::make_tuple(mstime_, i, c));
      }
      previousSpike[i] = t;
    }
  }
}

void LIFNetwork::updateIncomingWeights(int index) {
  // from spike in neuron index, find incoming connections. But since this
  // is only called when excitatory neurons spike, and the exc layer only
  // receives connections from input, so only look there
  for (size_t i = Nn; i < N; i++) {
    for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
      if ((*connectionTargets[i])[j] == index) {
        // connection j from neuron i to index
        float dv = connectionTrace[i-Nn];
        float dw = wmax - (*connectionWeights[i])[j];
        float update = stdp_lr * dv * pow(dw, 1.0);
        if (update > 0) {
          (*connectionWeights[i])[j] += update;
          if ((*connectionWeights[i])[j] > wmax) {
            (*connectionWeights[i])[j] = wmax;
          } else if ((*connectionWeights[i])[j] < wmin) {
            (*connectionWeights[i])[j] = wmin;
            // remove connection?
          }
        }
      }
    }
  }
}

void LIFNetwork::updateFromInput(int index) {
  // input neuron index spikes, we might have to negatively impact outgoing connections
  for (size_t i = 0; i < connectionTargets[index]->size(); i++) {
    float dv = postTrace[(*connectionTargets[index])[i]];
    float dw = wmax - (*connectionWeights[index])[i];
    float update = stdp_lr * dv * pow(dw, 1.0);
    if (update > 0) {
      (*connectionWeights[index])[i] -= update;
      if ((*connectionWeights[index])[i] > wmax) {
        (*connectionWeights[index])[i] = wmax;
      } else if ((*connectionWeights[index])[i] < wmin) {
        (*connectionWeights[index])[i] = wmin;
        // remove connection?
      }
    }
  }
}

void LIFNetwork::decayTrace() {
  for (size_t i = 0; i < Nd; i++) {
    if (connectionTrace[i] > 0) {
      connectionTrace[i] *= exp(-(t - previousSpike[i+Nn]) / tau_trace_pre);
    }
  }
  for (size_t i = 0; i < Ne; i++) {
    if (postTrace[i] > 0) {
      postTrace[i] *= exp(-(t - previousSpike[i]) / tau_trace_post);
    }
  }
}

void LIFNetwork::decayNeurons() {
  for (size_t i = 0; i < Nn; i++) {
    float diff = t - previousSpike[i];
    if (i < Ne) {
      // exc neuron
      S(0, i) *= exp(-diff / taue);
    } else {
      // inh neuron
      float diff = t - previousSpike[i];
      S(0, i) *= exp(-diff / taui);
    }
  }
}
void LIFNetwork::decayTheta() {
  for (size_t i = 0; i < Ne; i++) {
    float diff = t - previousSpike[i];
      thetas[i] *= exp(-diff / tau_theta);
  }
}

void LIFNetwork::cycle() {
  // Receive input spikes from image (or don't in an inactive cycle)
  presentData();

  // Add up spikes from the queue if the spike should be applied now
  for (size_t i = 0; i < Ne; i++) {
    processPreviousSpikes(i);
  }

  // saveStates();
  // Check whether a spike occurs in a neuron, and put that spike in the queue
  // at the given delay
  for (size_t i = 0; i < N; i++) {
    handleSpikes(i);
  }

  // Exponential decay on the neuron state
  decayNeurons();
  // Exponential decay on the weight traces
  if (learning) {
    decayTrace();
    decayTheta();
  }
}

void LIFNetwork::labelNeurons() {
  // For each neuron, count spikes per class
  int classSpikes[N][10];
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < 10; j++) {
      classSpikes[i][j] = 0;
    }
  }

  // Iterate through dataset once
  while (cur_img < label_limit) {
    cycle();
    t += dt;
    mstime_++;
    std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
              << (cur_img / float(label_limit))<< std::flush;
  }
  std::cout << '\n';
  std::cout << "No. of labeling spikes: " << firings.size() << '\n';

  for (size_t i = 0; i < firings.size(); i++) {
    classSpikes[std::get<1>(firings[i])][std::get<2>(firings[i])]++;
  }

  // For each neuron, if its response in this cycle was higher than the
  // previous highest, update the class associated with this neuron

  for (size_t i = 0; i < Nn; i++) {
    std::cout << "Spike count for neuron: " << i << ": { ";
    std::cout << classSpikes[i][0] << ", ";
    std::cout << classSpikes[i][1] << ", ";
    std::cout << classSpikes[i][2] << ", ";
    std::cout << classSpikes[i][3] << ", ";
    std::cout << classSpikes[i][4] << ", ";
    std::cout << classSpikes[i][5] << ", ";
    std::cout << classSpikes[i][6] << ", ";
    std::cout << classSpikes[i][7] << ", ";
    std::cout << classSpikes[i][8] << ", ";
    std::cout << classSpikes[i][9] << " } | Highest class is ";
    // std::cout << std::max_element(classSpikes[i], classSpikes[i]+10) - classSpikes[i] << '\n';
    // neuronClass[i] = std::max_element(classSpikes[i], classSpikes[i]+10) - classSpikes[i];
    int highest = 0;
    int c = 0;
    for (size_t j = 0; j < 10; j++) {
      if (classSpikes[i][j] >= highest) {
        highest = classSpikes[i][j];
        c = j;
      }
    }
    if (highest == 0) {
      // No activation at all in this neuron, so ignore it
      neuronClass[i] = -1;
    } else {
      neuronClass[i] = c;
    }
    std::cout << neuronClass[i] << " with: " << highest << '\n';
  }
}

int LIFNetwork::getLabelFromSpikes() {
  // spikes per neuron
  int neuronSpikes[N];
  // no. of neurons per class & spikes per class
  int classSpikes[10][2];
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 2; j++) {
      classSpikes[i][j] = 0;
    }
  }

  for (size_t i = 0; i < N; i++) {
    neuronSpikes[i] = 0;
  }
  // Active presentation of the image
  while (!sleepingCycle) {
    cycle();
    t += dt;
    mstime_++;
  }

  for (size_t i = 0; i < firings.size(); i++) {
    neuronSpikes[std::get<1>(firings[i])]++;
  }

  for (size_t i = 0; i < Nn; i++) {
    // label associated with this neuron
    int label = neuronClass[i];
    if (label == -1) {
      continue;
    }
    classSpikes[label][0]++;
    classSpikes[label][1] += neuronSpikes[i];
  }

  // default answer is 11, such that not coming up with a different answer
  // leads to a 0% accuracy
  float highest = 0.;
  int answer = 11;
  for (size_t i = 0; i < 10; i++) {
    float avg = classSpikes[i][1] / float(classSpikes[i][0]);
    std::cout << "Class: " << i << " spikes:" << classSpikes[i][1] << " Neurons: " << classSpikes[i][0] << " score: " << avg << '\n';
    if (avg > highest) {
        highest = avg;
        answer = i;
    }
  }
  firings.clear();
  return answer;
}

void LIFNetwork::plotSpikes() {
  for (auto spike : firings) {
    std::cerr << std::get<0>(spike) << ", " << std::get<1>(spike) << '\n';
  }
}

void LIFNetwork::plotWeights() {
  for (size_t i = Nn; i < N; i++) {
    for (size_t j = 0; j < connectionWeights[i]->size(); j++) {
      std::cerr << (*connectionWeights[i])[j] << '\n';
    }
  }
}

void LIFNetwork::saveWeights() {
  std::ofstream weightFile;
  weightFile.open ("weights.bin");
  for (size_t i = 0; i < N; i++) {
    weightFile << "Neuron: " << i << " weights: ";
    for (auto x : (*connectionWeights[i]) ) {
      weightFile << x << ", ";
    }
    weightFile << '\n';
  }
  weightFile.close();
}

void LIFNetwork::saveStates() {
  std::ofstream outfile("states.bin", std::ios_base::app);
  if (outfile.is_open()) {
    outfile << "Time: " << t;
    for (size_t i = 0; i < Ne; i++) {
      if (S(0,i) > 0.013) {
        outfile << " Neuron " << i << " state: " << S(0, i);
      }
    }
    outfile << "\n";
    outfile.close();
  }
}

void LIFNetwork::showWeightExtrema() {
  float highest = 0;
  float lowest = 1;
  size_t k,l,m,n;
  for (size_t i = Nn; i < N; i++) {
    for (size_t j = 0; j < connectionWeights[i]->size(); j++) {
      if ((*connectionWeights[i])[j] > highest) {
        highest = (*connectionWeights[i])[j];
        k = i;
        l = j;
      } else if ((*connectionWeights[i])[j] < lowest) {
        lowest = (*connectionWeights[i])[j];
        m = i;
        n = j;
      }
    }
  }
  std::cout << "Highest weight is " << k << " to " << (*connectionTargets[k])[l] << " with weight: " << highest << '\n';
  std::cout << "Lowest weight is " << m << " to " << (*connectionTargets[m])[n] << " with weight: " << lowest << '\n';
}
void LIFNetwork::showThetaExtrema() {
  float highest = 0;
  float lowest = 1;
  size_t k,l;
  for (size_t i = 0; i < thetas.size(); i++) {
    if (thetas[i] > highest) {
      highest = thetas[i];
      k = i;
    } else if (thetas[i] < lowest) {
      lowest = thetas[i];
      l = i;
    }
  }
  std::cout << "Highest theta is " << k << " with theta: " << highest << '\n';
  std::cout << "Lowest theta is " << l << " with theta: " << lowest << '\n';
}

void LIFNetwork::plotTrace() {
  for (size_t i = 0; i < Nd; i++) {
    std::cerr << mstime_ << ", " << i << ", " << connectionTrace[i] << '\n';
  }
}
void LIFNetwork::plotNeurons() {
  for (size_t i = 0; i < N; i++) {
    std::cerr << mstime_ << ", " << i << ", " << S(0, i) << '\n';
  }
}
