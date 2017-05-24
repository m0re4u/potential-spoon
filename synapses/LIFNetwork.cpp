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
  int i, j;

  // Update matrix
  A <<
      exp(-dt/taum), taue/(taum-taue)*(exp(-dt/taum)-exp(-dt/taue)), taui/(taum-taui)*(exp(-dt/taum)-exp(-dt/taui)),
      0, exp(-dt/taue), 0,
      0, 0, exp(-dt/taui);
  // state matrix
  S = Eigen::MatrixXd::Constant(3,N, 0);

  // Starting voltages
  for (i = 0; i < Ne; i++) {
    S(0,i) = -105.;
  }
  for (i = Ne; i < Nn; i++) {
    S(0,i) = -100.;
  }
  for (i = Nn; i < N; i++) {
    S(0,i) = 0; // does not matter, rate based
  }

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    refractory[i] = 0; // no neuron is has fired, so no refraction
    auto targets = new std::vector<int>();
    auto weights = new std::vector<float>();
    if (i < Ne) {
      // exc neuron connection to inhibitory: one on one
      targets->push_back(i+Ne);
      weights->push_back(10.4);
    } else if (i >= Ne && i < Nn) {
      // inh neuron connection to all exc except incoming
      for (j = 0; j < Ne; j++) {
        if (!(i - Ne == j)) {
          targets->push_back(j);
          weights->push_back(17);
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
  if (i < Ne) {
    if (refractory[i] > 0) {
      refractory[i]--;
      return;
    }
    if (S(0, i) > v_thresh_e) {
      // std::cout << "Spike in exc: " << i << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(1, j) += (*connectionWeights[i])[j];
      }
      S(0, i) = v_reset_e;  // reset potential
      refractory[i] = 50;
      // Store spike
      firings.push_back(std::make_tuple(mstime_, i));
    }
  } else if (i >= Ne && i < Nn) {
    if (refractory[i] > 0) {
      refractory[i]--;
      return;
    }
    if (S(0, i) > v_thresh_i) {
      // std::cout << "Spike in inh: " << i << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(2, j) += (*connectionWeights[i])[j];
      }
      refractory[i] = 20;
      S(0, i) = v_reset_i;  // reset potential
      // Store spike
      firings.push_back(std::make_tuple(mstime_, i));
    }
  } else {
    if (S(0, i) > v_thresh_i) { // does not matter, will fire
      // std::cout << "Spike in input" << '\n';
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        S(1, j) += (*connectionWeights[i])[j];
      }
      S(0, i) = v_reset_i;  // reset potential
      // Store spike
      firings.push_back(std::make_tuple(mstime_, i));

    }
  }
}

void LIFNetwork::cycle() {
  // Update potential
  auto result = A * S;
  S = result;

  // Store voltage of neuron 250 for plotting
  state.push_back(S(0, 250));
  // Receive input spikes from image (or don't in an inactive cycle)
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

void LIFNetwork::plotNeuron() {
  for (double volt : state) {
    std::cerr << volt << '\n';
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
