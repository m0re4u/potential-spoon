/**
 * Implements a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "LIFNetwork.h"
#include "CImg/CImg.h"

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

  // Variables for the active neurons in the second layer
  for (i = 0; i < Ne; i++) a[i] = 0.02; // excitatory
  for (i = Ne; i < Nn; i++) a[i] = 0.1;  // inhibitory

  for (i = 0; i < Ne; i++) d[i] = 8.0;  // excitatory
  for (i = Ne; i < Nn; i++) d[i] = 2.0;  // inhibitory

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
        }
      }
    } else {
      // input connections, all to all input to exc
      for (j = 0; j < Ne; j++) {
        post[i].push_back(j);
      }
    }
  }

  // Initialize connection weights
  for (i = 0; i < Ne; i++)  for (int j : post[i]) s[i].push_back(6.0);   // excitatory synaptic weights
  for (i = Ne; i < Nn; i++) for (int j : post[i]) s[i].push_back(-5.0);  // inhibitory synaptic weights
  // TODO: find the appropriate connection weight for the input -> exc layers
  for (i = Nn; i < N; i++)  for (int j : post[i]) s[i].push_back(1.0);  // input synaptic weights

  for (i = 0; i < N; i++)  for (int j : post[i]) sd[i].push_back(0.0);  // synaptic derivatives

  // Assign a delay to each neuron - all 1 ms at the moment
  for (i = 0; i < N; i++) {
    for (j = 0; j < D; j++) {
      delays_length[i][j] = 0;
    }
    for (k = 0; k < post[i].size(); k++) {
      // neuron i only has connections with delay 0 (1 ms)
      delays[i][0].push_back(k);
    }
  }

  // Set presynaptic information & weights
  for (i = 0; i < N; i++) {
    N_pre[i] = 0;
    for (j = 0; j < N; j++) {
      for (int k : post[j]) {
        if (post[j][k] == i) {
          // presynaptic connection from j --> i
          I_pre[i].push_back(j);
          for (dd = 0; dd < D; dd++) {  // 0 -> 20
            for (jj = 0; jj < delays_length[j][dd]; jj++) { // find the delay
              if (post[j][delays[j][dd][jj]] == i) {
                std::cout << "Neuron: " << j << " with delay: " << dd << "" << delays[j][dd][jj] << '\n';
                // D_pre[i][N_pre[i]] = dd;
              }
            }
          }
          s_pre[i]  = &s[j];  // pointer to the synaptic weight
          sd_pre[i] = &sd[j]; // pointer to the derivative
        }
      }
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < 1 + D; j++) {
      LTP[i][j] = 0.0;
    }
  }
  for (i = 0; i < N; i++)  LTD[i] = 0.0;
  for (i = 0; i < N; i++)  v[i] = -65.0;    // initial values for v
  for (i = 0; i < N; i++)  u[i] = 0.2*v[i]; // initial values for u
  //
  N_firings = 1;      // spike timings
  // firings[0][0] = -D; // put a dummy spike at -D for simulation efficiency
  // firings[0][1] = 0;  // index of the dummy spike
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

  img.display("First image");
}

bool LIFNetwork::generate_spike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4;
  double num = this->dist(this->gen);
  return num <= firing_rate * this->DT;
}

void LIFNetwork::input_spikes(unsigned image_index) {
  for (std::size_t i = 0, e = this->input_layer.size(); i != e; ++i) {
    bool spike = generate_spike(this->data[0][i]);
    if (spike) {
      std::cout << "Spike in neuron " << i << '\n';
      int x = i / 28;
      int y = i % 28;
      this->img_spikes[x][y]++;
    }
  }
}

void LIFNetwork::cycle() {
  size_t i, j, k;
  float  I[N];

  for (this->mstime_ = 0; this->mstime_ < 1; this->mstime_++) {
    std::cout << "Second: " << this->stime_ << " Cycle " << this->mstime_ << " working image: " << this->mstime_ / this->BOTH_TIME << '\n';

    for (i = 0; i < N; i++) {
      I[i] = 0.0; // reset input
    }

    for (i = 0; i < N; i++) {
      std::cout << "Voltage for neuron: " << i << " is: " << v[i] << '\n';
      if (v[i] >= 30) {  // did it fire?
        std::cout << "Firing neuron" << '\n';
        v[i] = -65.0;    // voltage reset
        u[i] += d[i];    // recovery variable reset
        LTP[i][this->mstime_ + D] = 0.1;
        LTD[i] = 0.12;
        // Loop over presynaptic connections
        // for (j = 0; j < N_pre[i]; j++) {
        //   // This spike was after pre-synaptic spikes, so strengthen the connection
        //   sd_pre[i][j] += LTP[I_pre[i][j]][this->mstime_ + D - D_pre[i][j] - 1];
        // }
        firings[N_firings  ][0] = this->mstime_;
        firings[N_firings++][1] = i;
        if (N_firings == N_firings_max) {
          std::cout << "Two many spikes at t=" << this->mstime_ << " (ignoring all)";N_firings=1;
        }
      }
    }
    std::cout << "Firings: " << N_firings << '\n';
    // k = N_firings;
    // while (this->mstime_-firings[--k][0] <D) {
    //   for (j = 0; j < delays_length[firings[k][1]][this->mstime_ - firings[k][0]]; j++) {
    //     i = post[firings[k][1]][delays[firings[k][1]][this->mstime_ - firings[k][0]][j]];
    //     I[i] += s[firings[k][1]][delays[firings[k][1]][this->mstime_ - firings[k][0]][j]];
    //     if (firings[k][1] <Ne) // this spike is before postsynaptic spikes
    //       sd[firings[k][1]][delays[firings[k][1]][this->mstime_ - firings[k][0]][j]] -= LTD[i];
    //   }
    // }
    // for (i=0;i<N;i++) {
    //   v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // for numerical stability
    //   v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // time step is 0.5 ms
    //   u[i]+=a[i]*(0.2*v[i]-u[i]);
    //   LTP[i][this->mstime_+D+1]=0.95*LTP[i][this->mstime_+D];
    //   LTD[i]*=0.95;
    // }

    // if (this->sleepingCycle) {
    //   // Check state of the next cycle
    //   this->cycle_switcher++;
    //   if (this->cycle_switcher >= this->SLEEP_TIME) {
    //     this->sleepingCycle = false;
    //     this->cycle_switcher = 0;
    //   }
    // } else {
    //   // this->input_spikes(this->mstime_ / this->BOTH_TIME);
    //   // Check state of the next cycle
    //   this->cycle_switcher++;
    //   if (this->cycle_switcher >= this->IMG_TIME) {
    //     this->cycle_switcher = 0;
    //     this->sleepingCycle = true;
    //   }
    // }
  }
}

void LIFNetwork::prepare() {
  // size_t i, j, k;
  // for (i=0;i<N;i++)    // prepare for the next sec
  //   for (j=0;j<D+1;j++)
  //   LTP[i][j]=LTP[i][1000+j];
  // k=N_firings-1;
  // while (1000-firings[k][0]<D) k--;
  // for (i=1;i<N_firings-k;i++)
  // {
  //   firings[i][0]=firings[k+i][0]-1000;
  //   firings[i][1]=firings[k+i][1];
  // }
  // N_firings = N_firings-k;
  //
  // for (i=0;i<Ne;i++)  // modify only exc connections
  // for (j=0;j<M;j++)
  // {
  //   s[i][j]+=0.01+sd[i][j];
  //   sd[i][j]*=0.9;
  //   if (s[i][j]>sm) s[i][j]=sm;
  //   if (s[i][j]<0) s[i][j]=0.0;
  // }
}
