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

  // Indexing variables
  int i, j, k, jj, dd;

  A <<
      exp(-dt/taum), taue/(taum-taue)*(exp(-dt/taum)-exp(-dt/taue)), taui/(taum-taui)*(exp(-dt/taum)-exp(-dt/taui)),
      0, exp(-dt/taue), 0,
      0, 0, exp(-dt/taui);

  S = Eigen::MatrixXd::Constant(3,N, 0);
  for (i = 0; i < N; i++) {
    float val = (this->dist1(this->gen) * (vt - vr)) + vr;
    S(0,i) = val;
  }

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    auto targets = new std::vector<int>();
    auto weights = new std::vector<float>();
    if (i < Ne) {
      // exc neuron connection to inhibitory: one on one
      targets->push_back(i+Ne);
      weights->push_back(1.62);
    } else if (i >= Ne && i < Nn) {
      // inh neuron connection to all exc except incoming
      for (j = 0; j < Ne; j++) {
        if (!(i - Ne == j)) {
          targets->push_back(j);
          weights->push_back(-9);
        }
      }
    } else {
      // input connections, all to all input to exc
      for (j = 0; j < Ne; j++) {
        targets->push_back(j);
        float val = this->dist_weights(this->gen);
        weights->push_back(val);
      }
    }
    connectionTargets.push_back(targets);
    connectionWeights.push_back(weights);
  }
}

void LIFNetwork::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) {
  this->data = dataset;
  this->labels = labels;
}

bool LIFNetwork::generateSpike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4000.; // per millisecond
  double num = this->dist1(this->gen);
  return num <= firing_rate;
}

void LIFNetwork::inputSpikes() {
  for (int i = Nn; i != N; ++i) {
    assert(i - Nd > 0);
    bool spike = generateSpike(this->data[this->cur_img][i - Nd]);
    if (spike) {
      S(0, i) = 1;
    }
  }
}

void LIFNetwork::presentData() {
  if (sleepingCycle) {
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= SLEEP_TIME) {
      cur_img++;
      sleepingCycle = false;
      cycle_switcher = 0;
    }

  } else {
    inputSpikes();
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= IMG_TIME) {
      cycle_switcher = 0;
      sleepingCycle = true;
    }
  }
}

void LIFNetwork::handleSpikes(int i) {
  bool spiked = false;
  if (i < Ne) {
    if (S(0, i) > vt) {
      // std::cout << "Spike in exc: " << i << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(1, j) += (*connectionWeights[i])[j];
      }
      spiked = true;
    }
  } else if (i >= Ne && i < Nn) {
    if (S(0, i) > vt) {
      // std::cout << "Spike in inh: " << i << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(2, j) += (*connectionWeights[i])[j];
      }
      spiked = true;
    }
  } else {
    if (S(0, i) > vt) {
      // std::cout << "Spike in input" << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(1, j) += (*connectionWeights[i])[j];
      }
      spiked = true;
    }
  }
  if (spiked) {
    S(0, i) = vr;  // reset potential
    // Store spike
    firings.push_back(std::make_tuple(mstime_, i));
  }
}

void LIFNetwork::cycle() {
  // Update potential
  auto result = A * S;
  S = result;

  // Store voltage of neuron 0 for plotting
  state.push_back(S(0, 0));
  // Receive input spikes from image
  presentData();
  // Handle the spikes that occur after updating voltages
  for (size_t i = 0; i < N; i++) {
    handleSpikes(i);
  }
}
void LIFNetwork::labelNeurons() {
  int cycleSpikes[N];
  for (size_t i = 0; i < N; i++) {
    cycleSpikes[i] = 0;
  }

  cycle();
  for (size_t i = 0; i < firings.size(); i++) {
    for (size_t j = 0; j < N; j++) {
      if (std::get<1>(firings[i]) == j) {
        // count number of spikes per neuron
        cycleSpikes[j]++;
      }
    }
  }
  // For each neuron, if its response in this cycle was higher than the
  // previous highest, update the class associated with this neuron
  for (size_t i = 0; i < N; i++) {
    if (cycleSpikes[i] > highestSpikes[i][0]) {
      highestSpikes[i][0] = cycleSpikes[i];
      highestSpikes[i][1] = int(this->labels[cur_img]);
    }
  }
}

int LIFNetwork::getLabelFromSpikes() {
  int cycleSpikes[N];
  int classSpikes[10][2];

  for (size_t i = 0; i < N; i++) {
    cycleSpikes[i] = 0;
  }

  cycle();
  for (size_t i = 0; i < firings.size(); i++) {
    for (size_t j = 0; j < N; j++) {
      if (std::get<1>(firings[i]) == j) {
        cycleSpikes[j]++;
      }
    }
  }
  for (size_t i = 0; i < N; i++) {
    // label associated with this neuron
    int label = highestSpikes[i][1];
    classSpikes[label][0]++;
    classSpikes[label][1] += cycleSpikes[i];
  }

  float highest = 0.;
  // default answer is 11, such that not coming up with a different answer
  // leads to a 0% accuracy
  int answer = 11;
  for (size_t i = 0; i < 10; i++) {
    float avg = classSpikes[i][1] / float(classSpikes[i][0]);
    if (avg > highest) {
        highest = avg;
        answer = i;
    }
  }
  return answer;
}

void LIFNetwork::plotSpikes() {
  std::cout << "#spikes: " << firings.size() << '\n';
  for (auto spike : firings) {
    std::cerr << std::get<0>(spike) << ", " << std::get<1>(spike) << '\n';
  }
}

// int main(int argc, char const *argv[]) {
//   LIFNetwork *n = new LIFNetwork();
//   std::cout << "Reading in MNIST dataset.." << '\n';
//   auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
//   n->load_dataset(dataset.training_images, dataset.training_labels);
//   std::cout << "Init params" << '\n';
//   n->initialize_params();
//
//   for (n->mstime_ = 0; n->mstime_ < 1000; n->mstime_++) {
//     std::cout << "Cycle: " << n->mstime_ << '\n';
//     n->cycle();
//   }
//   n->plotSpikes();
//
//   return 0;
// }
