/**
 * Implements a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "COMPlifnetwork.h"

void COMPlifnetwork::initialize_params(json config) {
  this->config = config;
  // random number generation
  std::mt19937 generator(this->rd());
  this->gen = generator;
  std::uniform_real_distribution<> distribution1(0, 1);
  this->dist1 = distribution1;
  std::uniform_real_distribution<> distribution_weights(0, config["weight_max"]);
  this->dist_weights = distribution_weights;

  // Indexing variables
  int i, j, k, jj, dd;

  // Variables for the active neurons
  for (i = 0; i < Ne; i++) a[i] = 0.02; // excitatory
  for (i = Ne; i < Nn; i++) a[i] = 0.1;  // inhibitory
  for (i = Nn; i < N; i++) a[i] = 0.02;  // input

  for (i = 0; i < Ne; i++) d[i] = 8.0;  // excitatory
  for (i = Ne; i < Nn; i++) d[i] = 2.0;  // inhibitory
  for (i = Nn; i < N; i++) d[i] = 8.0;  // input

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    if (i < Ne) {
      // exc neuron connection to inhibitory: one on one
      post[i].push_back(i+Ne);
    } else if (i >= Ne && i < Nn) {
      // inh neuron connection to all exc except incoming
      for (j = 0; j < Ne; j++) {
        if (!(i - Ne == j)) {
          post[i].push_back(j);
        } else {
          // No connection to presynaptic exc neuron, instead assign an impossible neuron
          post[i].push_back(-1);
        }
      }
    } else {
      // input connections, all to all input to exc
      for (j = 0; j < Ne; j++) {
        post[i].push_back(j);
      }
    }
  }

  std::vector<std::vector<float>*> s_tmp;
  s_tmp.reserve(10);
  std::vector<std::vector<float>*> sd_tmp;
  sd_tmp.reserve(10);
  // Initialize connection weights
  for (i = 0; i < Ne; i++)  {
    auto v = new std::vector<float>();
    for (int j : post[i]) v->push_back(config["e_weights"]);   // excitatory synaptic weights
    s_tmp.push_back(v);
  }
  for (i = Ne; i < Nn; i++) {
    auto v = new std::vector<float>();
    for (int j : post[i]) v->push_back(config["i_weights"]);  // inhibitory synaptic weights
    s_tmp.push_back(v);
  }
  for (i = Nn; i < N; i++) {
    auto v = new std::vector<float>();
    for (int j : post[i]) {
      float val = this->dist_weights(this->gen);
      v->push_back(val);  // input synaptic weights
    }
    s_tmp.push_back(v);
  }

  for (i = 0; i < N; i++) {
    auto v = new std::vector<float>();
    for (int j : post[i]) v->push_back(0.0);  // synaptic derivatives
    sd_tmp.push_back(v);
  }
  this->s = s_tmp;
  this->sd = sd_tmp;

  // Assign a delay to each neuron - all 1 ms at the moment
  for (i = 0; i < N; i++) {
    short ind = 0;
    for (j = 0; j < D; j++) {
      delays_length[i][j] = 0;
    }

    delays_length[i][0] = post[i].size();
    for (k = 0; k < post[i].size(); k++) {
      // neuron i only has connections with delay 0 (1 ms), so push back the
      // connection at index ind as having delay 0
      delays[i][0].push_back(ind++);
    }
  }

  // Set presynaptic information & weights
  for (i = 0; i < N; i++) {
    N_pre[i] = 0;
    std::vector<float>* pre_v = new std::vector<float>();
    std::vector<float>* pre_vd = new std::vector<float>();
    for (j = 0; j < N; j++) {
      for (int k = 0; k < post[j].size(); k++) {
        if (post[j][k] == i) {
          // presynaptic connection from j --> i
          I_pre[i].push_back(j);
          for (dd = 0; dd < D; dd++) {  // 0 -> 20
            for (jj = 0; jj < delays_length[j][dd]; jj++) { // find the delay
              if (post[j][delays[j][dd][jj]] == i) {
                D_pre[i].push_back(dd);
              }
            }
          }
          pre_v->push_back(s[j]->at(k));   // pointer to the synaptic weight
          pre_vd->push_back(sd[j]->at(k)); // pointer to the derivative
          N_pre[i]++;              // Count this as a presynaptic connection
        }
      }
    }
    s_pre.push_back(pre_v);
    sd_pre.push_back(pre_vd);

  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < 1 + D; j++) {
      LTP[i][j] = 0.0; // initial values for STDP functions
    }
  }
  for (i = 0; i < N; i++)  LTD[i] = 0.0;               // initial values for STDP function
  for (i = 0; i < Ne; i++)  v[i] = config["e_init_v"]; // initial values for v for exc neurons
  for (i = Ne; i < N; i++)  v[i] = config["i_init_v"]; // initial values for v for inh neurons
  // value for input neurons doesnt matter, also assigned i_init_v
  for (i = 0; i < N; i++) {
    u[i] = 0.2 * v[i];       // initial values for u
    highestSpikes[i][0] = 0; // highest spike count
    highestSpikes[i][1] = 0; // highest spike label
  }
  firings.reserve(N_firings_max);
  N_firings = 0; // number of spikes in a second
}

void COMPlifnetwork::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) {
  this->data = dataset;
  this->labels = labels;
}

void COMPlifnetwork::show_image(std::vector<unsigned char, std::allocator<unsigned char>> &vec) {
  cimg_library::CImg<uint8_t> img(28,28,1,1);
  unsigned pixel_ = 0;
  cimg_forXY(img,x,y) {  // Do 2 nested loops
    pixel_ = y * 28 + x;
    img(x,y) = vec[pixel_];
  }
  img.display("Test image");
}

bool COMPlifnetwork::generate_spike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4000.; // per millisecond
  double num = this->dist1(this->gen);
  return num <= firing_rate;
}

void COMPlifnetwork::input_spikes() {
  for (int i = Nn; i != N; ++i) {
    assert(i - 784 > 0);
    bool spike = generate_spike(this->data[this->cur_img][i - 784]);
    if (spike) {
      v[i] = 50;
    }
  }
}

void COMPlifnetwork::presentData() {
  if (this->sleepingCycle) {
    // Check state of the next cycle
    this->cycle_switcher++;
    if (this->cycle_switcher >= this->SLEEP_TIME) {
      this->cur_img++;
      this->sleepingCycle = false;
      this->cycle_switcher = 0;
    }

  } else {
    this->input_spikes();
    // Check state of the next cycle
    this->cycle_switcher++;
    if (this->cycle_switcher >= this->IMG_TIME) {
      this->cycle_switcher = 0;
      this->sleepingCycle = true;
    }
  }
}

void COMPlifnetwork::handleSpikes(int index, bool learning) {
  bool spiked = false;
  bool noninput = false;
  if (index >= Nn) {
    // input neurons
    if (v[index] >= 30) {  // did it fire?
      v[index] = -65.0;    // input voltage does not matter
      // std::cout << "Spike in input: " << index << '\n';
      spiked = true;
    }
  } else if (index >= Ne && index < Nn) {
    // inh neurons
    if (v[index] >= config["i_thresh"]) {  // did it fire?
      v[index] = config["i_reset"];    // voltage reset
      // std::cout << "Spike in inh: " << index << '\n';
      spiked = true;
      noninput = true;
    }
  } else {
    // exc neurons
    if (v[index] >= config["e_thresh"]) {  // did it fire?
      v[index] = config["e_reset"]; // voltage reset
      // std::cout << "Spike in exc: " << index << '\n';
      spiked = true;
      noninput = true;
    }
  }

  if (spiked) {
    // uncomment to see spike pattern
    std::cerr << ((this->stime_*1000) + this->mstime_) << ", " << index << '\n';
    u[index] += d[index]; // recovery variable reset
    if (noninput && learning) {
      // STDP function variables
      LTP[index][this->mstime_ + D] = 0.1;
      LTD[index] = 0.12;
      // Loop over presynaptic connections
      for (int j = 0; j < N_pre[index]; j++) {
        // This spike was after pre-synaptic spikes, so strengthen the connection
        (*sd_pre[index])[j] += LTP[I_pre[index][j]][this->mstime_ + D - D_pre[index][j] - 1];
      }
    }
    // Store the firings
    firings.push_back(std::make_tuple(this->mstime_, index));
    N_firings++;
    if (N_firings == N_firings_max) {
      std::cout << "Too many spikes at t=" << this->mstime_ << " (ignoring all)";N_firings=1;
    }
  }
}

void COMPlifnetwork::prepare(bool learning) {
  size_t i, j, k;
  if (learning) {
    for (i = 0; i < N; i++) { // prepare for the next sec
      for (j = 0; j < D + 1; j++) {
        LTP[i][j] = LTP[i][1000+j];
      }
    }
  }

  k = N_firings - 1;
  while (1000-std::get<0>(firings[k]) < D) k--;
  for (i = 1; i < N_firings - k; i++) {
    std::get<0>(firings[i]) = std::get<0>(firings[k+i]) - 1000;
    std::get<1>(firings[i]) = std::get<1>(firings[k+i]);
  }
  N_firings = N_firings - k;

  if (learning) {
    // Use derivatives to update the weight
    for (i = 0; i < Ne; i++) {
      // modify only exc connections
      for (j = 0; j < post[i].size();  j++) {
        (*s[i])[j]  += 0.01 + (*sd[i])[j];
        (*sd[i])[j] *= 0.9;
        // cap weights
        if ((*s[i])[j] > sm) (*s[i])[j] = sm;
        if ((*s[i])[j] < 0)  (*s[i])[j] = 0.0;
      }
    }
  }
}

void COMPlifnetwork::processDelayedSpikes(int k, float inputCurrents[], bool learning) {
  for (int j = 0; j < delays_length[std::get<1>(firings[k])][this->mstime_ - std::get<0>(firings[k])]; j++) {
    // find the neuron from first index using connection second index
    int i = post[std::get<1>(firings[k])][delays[std::get<1>(firings[k])][this->mstime_ - std::get<0>(firings[k])][j]];
    if (i == -1) {
      continue;
    }
    // Add the weight of that connection as a delayed spike
    inputCurrents[i] += (*s[std::get<1>(firings[k])])[delays[std::get<1>(firings[k])][this->mstime_ - std::get<0>(firings[k])][j]];
    if (std::get<1>(firings[k]) < Ne && learning) { // this is an excitatory spike
      // Update derivative of the connection weight
      (*sd[std::get<1>(firings[k])])[delays[std::get<1>(firings[k])][this->mstime_ - std::get<0>(firings[k])][j]] -= LTD[i];
    }
  }
}

void COMPlifnetwork::updatePotential(int i, float inputCurrent) {
  v[i] += 0.5 * ((0.04*v[i]+5)*v[i]+140-u[i]+inputCurrent); // for numerical stability
  v[i] += 0.5 * ((0.04*v[i]+5)*v[i]+140-u[i]+inputCurrent); // time step is 0.5 ms
  u[i] += a[i]* (0.2*v[i]-u[i]);
  LTP[i][this->mstime_+D+1] = 0.95*LTP[i][this->mstime_+D];
  LTD[i] *= 0.95;
}

void COMPlifnetwork::cycle(bool learning) {
  size_t i, j;
  int k;
  float  I[N];

  for (this->mstime_ = 0; this->mstime_ < 500; this->mstime_++) {
    // uncomment to see voltage for a neuron
    // std::cerr << (this->stime_*500)+this->mstime_ << " " << v[0] << '\n';

    for (i = 0; i < N; i++) I[i] = 0.0; // reset input
    this->presentData();
    for (i = 0; i < N; i++) {
      this->handleSpikes(i, learning);
    }

    k = N_firings-1;
    // While the difference between an observed spike and the current timestamp
    // is smaller than the maximum spike delay, process spikes
    while ((this->mstime_ - std::get<0>(firings[k])) < D && k >= 0) {
      this->processDelayedSpikes(k, I, learning);
      k--;
    }

    // Update potential for every neuron using the processed spikes
    for (i = 0; i < N; i++) {
      this->updatePotential(i, I[i]);
    }
  }
}

void COMPlifnetwork::labelNeurons() {
  int cycleSpikes[N];
  for (size_t i = 0; i < N; i++) {
    cycleSpikes[i] = 0;
  }

  this->cycle(false);
  std::cout << "firing rate=" << float(N_firings)/N << "\n";
  for (size_t i = 0; i < N_firings; i++) {
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

  this->prepare(false);
}

int COMPlifnetwork::getLabelFromSpikes() {
  int cycleSpikes[N];
  int classSpikes[10][2];

  for (size_t i = 0; i < N; i++) {
    cycleSpikes[i] = 0;
  }

  this->cycle(false);
  for (size_t i = 0; i < N_firings; i++) {
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
    // std::cout << "Class " << i << " has activation: " << avg << '\n';
    if (avg > highest) {
        highest = avg;
        answer = i;
    }
  }
  this->prepare(false);

  return answer;
}
