/**
 * Implements a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "LIFNetwork.h"

#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1

LIFNetwork::LIFNetwork() {
  std::mt19937 generator(this->rd());
  this->gen = generator;
  std::uniform_real_distribution<> distribution(0, 1);
  this->dist = distribution;
}

void LIFNetwork::initialize_params() {
  std::cout << "Initializing paramters" << '\n';
  // Indexing variables
  int i, j, k, jj, dd, exists, r;

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
    for (int j : post[i]) v->push_back(6.0);   // excitatory synaptic weights
    s_tmp.push_back(v);
  }
  for (i = Ne; i < Nn; i++) {
    auto v = new std::vector<float>();
    for (int j : post[i]) v->push_back(-5.0);  // inhibitory synaptic weights
    s_tmp.push_back(v);
  }
  // TODO: find the appropriate connection weight for the input -> exc layers
  for (i = Nn; i < N; i++) {
    auto v = new std::vector<float>();
    for (int j : post[i]) v->push_back(1.);  // input synaptic weights
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
              if (delays[j][dd][jj] == i) {
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
      LTP[i][j] = 0.0;
    }
  }
  for (i = 0; i < N; i++)  LTD[i] = 0.0;
  for (i = 0; i < Ne; i++)  v[i] = -65.0;    // initial values for v for exc neurons
  for (i = Ne; i < N; i++)  v[i] = -60.0;    // initial values for v for inh neurons
  // value for input neurons doesnt matter
  for (i = 0; i < N; i++)  u[i] = 0.2*v[i]; // initial values for u
  //
  N_firings = 0;      // number of spikes in a second
  std::cout << "Finished initializing paramters" << '\n';
}

void LIFNetwork::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset) {
  this->data = dataset;
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

bool LIFNetwork::generate_spike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4000.; // per millisecond
  double num = this->dist(this->gen);
  return num <= firing_rate;
}

void LIFNetwork::input_spikes() {
  for (int i = Nn; i != N; ++i) {
    assert(i - 784 > 0);
    bool spike = generate_spike(this->data[this->cur_img][i - 784]);
    if (spike) {
      v[i] = 50;
    }
  }
}

void LIFNetwork::present_data() {
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

void LIFNetwork::handleSpikes(int index, bool learning) {
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
    if (v[index] >= -40) {  // did it fire?
      v[index] = -45.0;    // voltage reset
      // std::cout << "Spike in inh: " << index << '\n';
      spiked = true;
      noninput = true;
    }
  } else {
    // exc neurons
    if (v[index] >= -52) {  // did it fire?
      v[index] = -65.0;    // voltage reset
      // std::cout << "Spike in exc: " << index << '\n';
      spiked = true;
      noninput = true;
    }
  }

  if (spiked) {
    if (noninput && learning) {
      u[index] += d[index];    // recovery variable reset
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
    firings[N_firings  ][0] = this->mstime_;
    firings[N_firings++][1] = index;
    if (N_firings == N_firings_max) {
      std::cout << "Too many spikes at t=" << this->mstime_ << " (ignoring all)";N_firings=1;
    }
  }
}

void LIFNetwork::prepare() {
  size_t i, j, k;
  for (i = 0; i < N; i++) { // prepare for the next sec
    for (j = 0; j < D + 1; j++) {
      LTP[i][j] = LTP[i][1000+j];
    }
  }
  k = N_firings - 1;
  while (1000-firings[k][0] < D) k--;
  for (i = 1; i < N_firings - k; i++) {
    firings[i][0] = firings[k+i][0] - 1000;
    firings[i][1] = firings[k+i][1];
  }
  N_firings = N_firings-k;

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

void LIFNetwork::processDelayedSpikes(int k, float inputCurrents[]) {
  for (int j = 0; j < delays_length[firings[k][1]][this->mstime_ - firings[k][0]]; j++) {
    // find the neuron from first index using connection second index
    int i = post[firings[k][1]][delays[firings[k][1]][this->mstime_ - firings[k][0]][j]];
    // Add the weight of that connection as a delayed spike
    inputCurrents[i] += (*s[firings[k][1]])[delays[firings[k][1]][this->mstime_ - firings[k][0]][j]];
    if (firings[k][1] < Ne) { // this is an excitatory spike
      // Update derivative of the connection weight
      (*sd[firings[k][1]])[delays[firings[k][1]][this->mstime_ - firings[k][0]][j]] -= LTD[i];
    }
  }
}

void LIFNetwork::updatePotential(int i, float inputCurrent) {
  v[i] += 0.5 * ((0.04*v[i]+5)*v[i]+140-u[i]+inputCurrent); // for numerical stability
  v[i] += 0.5 * ((0.04*v[i]+5)*v[i]+140-u[i]+inputCurrent); // time step is 0.5 ms
  u[i] += a[i]* (0.2*v[i]-u[i]);
  LTP[i][this->mstime_+D+1] = 0.95*LTP[i][this->mstime_+D];
  LTD[i] *= 0.95;
}

void LIFNetwork::cycle() {
  size_t i, j;
  int k;
  float  I[N];

  for (this->mstime_ = 0; this->mstime_ < 500; this->mstime_++) {
    std::cerr << std::setfill('0') << std::setw(5) << this->stime_ <<this->mstime_ << " " << v[0] << '\n';
    // std::cout << "Second: " << this->stime_ << " Cycle " << this->mstime_ << " working image: " << this->mstime_ / this->BOTH_TIME << '\n';

    for (i = 0; i < N; i++) I[i] = 0.0; // reset input
    this->present_data();
    for (i = 0; i < N; i++) {
      this->handleSpikes(i, true);
    }

    k = N_firings;
    // While the difference between an observed spike and the current timestamp
    // is smaller than the maximum spike delay, process spikes
    while ((this->mstime_ - firings[--k][0]) < D) {
      this->processDelayedSpikes(k, I);
    }

    // Update potential for every neuron using the processed spikes
    for (i = 0; i < N; i++) {
      this->updatePotential(i, I[i]);
    }
  }
}
