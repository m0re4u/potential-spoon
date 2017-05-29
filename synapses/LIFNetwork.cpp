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
    for (j = 0; j < max_delay; j++) {
      spikeQueue[j][i] = 0;
    }
  }
  for (i = Ne; i < Nn; i++) {
    S(0,i) = -100.;
  }
  for (i = Nn; i < N; i++) {
    S(0,i) = -100; // does not matter, rate based
  }

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    refractory[i] = 0; // no neuron is has fired, so no refraction
    auto targets = new std::vector<int>();
    auto delays = new std::vector<int>();
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
        int d = this->dist_delay(this->gen);
        delays->push_back(d);
        float val = this->dist_weights(this->gen);
        weights->push_back(val);
      }
    }
    connectionTargets.push_back(targets);
    connectionDelays.push_back(delays);
    connectionWeights.push_back(weights);
  }
}

void LIFNetwork::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) {
  // reset simulation variables
  this->mstime_ = 0;
  this->cycle_switcher = 0;
  this->cur_img = 0;
  this->sleepingCycle = false;
  this->data = dataset;
  this->labels = labels;
  firings.clear();
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
    assert(i - Nn >= 0);
    bool spike = generateSpike(this->data[this->cur_img][i - Nn]);
    if (spike) {
      S(0, i) = 111;
    } else {
      S(0, i) = -60;
    }
  }
}

void LIFNetwork::presentData() {
  if (sleepingCycle) {
    for (size_t i = Nn; i < N; i++) {
      S(0, i) = -60;
    }
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
    // Add up any delayed spikes
    if (spikeQueue[mstime_% max_delay][i] > 0) {
      S(0, i) += spikeQueue[mstime_ % max_delay][i];
      spikeQueue[mstime_ % max_delay][i] = 0;
    }
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
      int c = int(this->labels[this->cur_img]);
      previousSpike[i] = t;
      firings.push_back(std::make_tuple(mstime_, i, c));
    }
  } else if (i >= Ne && i < Nn) {
    // Add up any delayed spikes
    if (spikeQueue[mstime_% max_delay][i] > 0) {
      S(0, i) += spikeQueue[mstime_ % max_delay][i];
      spikeQueue[mstime_ % max_delay][i] = 0;
    }
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
      int c = int(this->labels[this->cur_img]);
      firings.push_back(std::make_tuple(mstime_, i, c));
    }
  } else {
    if (S(0, i) > v_thresh_i) { // does not matter, will fire
      for (size_t j = 0; j < connectionTargets[i]->size(); j++) {
        spikeQueue[(mstime_ + (*connectionDelays[i])[j]) % max_delay][j] += (*connectionWeights[i])[j];
      }
      S(0, i) = v_reset_i;  // reset potential
      // Store spike
      int c = int(this->labels[this->cur_img]);
      firings.push_back(std::make_tuple(mstime_, i, c));

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
  // reset image counter
  this->cur_img = 0;

  // For each neuron, count spikes per class
  int classSpikes[N][10];
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < 10; j++) {
      classSpikes[i][j] = 0;
    }
  }
  firings.clear();

  // Iterate through dataset once
  for (size_t i = 0; i < 60000; i++) {
    cycle();
    std::cout << "Label iteration: " << i << " class: " << int(this->labels[this->cur_img]) << "\n";
  }
  std::cout << "No. of labeling spikes: " << firings.size() << '\n';

  int lastClass = 0;
  for (size_t i = 0; i < firings.size(); i++) {
    if (lastClass != std::get<2>(firings[i])) {
      std::cout << "Class: " << std::get<2>(firings[i]) << '\n';
      lastClass = std::get<2>(firings[i]);
    }
    classSpikes[std::get<1>(firings[i])][std::get<2>(firings[i])]++;
  }

  // For each neuron, if its response in this cycle was higher than the
  // previous highest, update the class associated with this neuron
  for (size_t i = 0; i < N; i++) {
    // std::cout << "Spike count for neuron: " << i << ": { ";
    // std::cout << classSpikes[i][0] << ", ";
    // std::cout << classSpikes[i][1] << ", ";
    // std::cout << classSpikes[i][2] << ", ";
    // std::cout << classSpikes[i][3] << ", ";
    // std::cout << classSpikes[i][4] << ", ";
    // std::cout << classSpikes[i][5] << ", ";
    // std::cout << classSpikes[i][6] << ", ";
    // std::cout << classSpikes[i][7] << ", ";
    // std::cout << classSpikes[i][8] << ", ";
    // std::cout << classSpikes[i][9] << " } | Highest class is ";
    // std::cout << std::max_element(classSpikes[i], classSpikes[i]+10) - classSpikes[i] << '\n';
    neuronClass[i] = std::max_element(classSpikes[i], classSpikes[i]+10) - classSpikes[i];
  }
}

int LIFNetwork::getLabelFromSpikes() {
  // spikes per neuron
  int neuronSpikes[N];
  // no. of neurons per class & spikes per class
  int classSpikes[10][2];
  // reset firings
  firings.clear();

  for (size_t i = 0; i < N; i++) {
    neuronSpikes[i] = 0;
  }

  // 350 ms input test image + 150 ms sleep
  for (size_t i = 0; i < 500; i++) {
    cycle();
  }

  for (size_t i = 0; i < firings.size(); i++) {
    neuronSpikes[std::get<1>(firings[i])]++;
  }

  for (size_t i = 0; i < N; i++) {
    // label associated with this neuron
    int label = neuronClass[i];
    classSpikes[label][0]++;
    classSpikes[label][1] += neuronSpikes[i];
  }

  // default answer is 11, such that not coming up with a different answer
  // leads to a 0% accuracy
  float highest = 0.;
  int answer = 11;
  for (size_t i = 0; i < 10; i++) {
    float avg = classSpikes[i][1] / float(classSpikes[i][0]);
    if (avg > highest) {
        highest = avg;
        answer = i;
    }
    // std::cout << classSpikes[i][1] << " | ";
  }
  // std::cout << '\n';
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
