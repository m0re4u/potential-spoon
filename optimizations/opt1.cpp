/**
 * Implements a network of LIF neurons
 * @author Michiel van der Meer <michiel@dutchnaoteam.nl>
 */

#include "opt1.h"

void Opt1Network::initialize_params() {
  // random number seed
  srand (static_cast <unsigned> (time(0)));

  // Indexing variables
  int i, j;

  // Reset the values used as states, spike queue, and refractory counters
  reset_values();
  for (i = 0; i < Ne; i++) {
    for (j = 0; j < Nd; j++) {
      incomingWeights[i][j] = 0;
    }
  }

  // Create connections between neurons
  for (i = 0; i < N; i++) {
    S[i]= 0.0; // initial state
    if (i < Ne) {
      // exc neuron connection to inhibitory: one on one
      excTargets[i][0] = i+Ne;
      excWeights[i][0] = 0.026;
    } else if (i >= Ne && i < Nn) {
      // inh neuron connection to all exc except incoming
      int k = 0;
      for (j = 0; j < Ne; j++) {
        if (!(i - Ne == j)) {
          inhTargets[i-Ne][k] = j;
          inhWeights[i-Ne][k] = -0.03;
          k++;
        }
      }
    } else {
      // input connections, all to all input to exc
      for (j = 0; j < Ne; j++) {
        inputTargets[i-Nn][j] = j;
        int d = (rand() % static_cast<int>(max_delay + 1));
        inputDelays[i-Nn][j] = d;
        // was 0.0001
        float val = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 0.0001));
        inputWeights[i-Nn][j] = val;
      }
      connectionTrace[i-Nn] = 0.;
    }
  }
  for (i = 0; i < Ne; i++) {
    thetas[i] = 0.0;
    for (size_t j = 0; j < Nd; j++) {
      for (size_t k = 0; k < Ne; k++) {
        if ( inputTargets[j][k] == i) {
          incomingWeights[i][j] = &(inputWeights[j][i]);
        }
      }
    }
  }
}

void Opt1Network::reset_values() {
  // Starting voltages
  for (size_t i = 0; i < Ne; i++) {
    S[i] = 0;
    // Spike delays only occur in input to exc connections - reset them
    for (size_t j = 0; j < max_delay; j++) {
      spikeQueue[j][i] = 0; // very cache inefficient :/
    }
  }
  for (size_t i = Ne; i < Nn; i++) {
    S[i] = 0;
  }
  for (size_t i = Nn; i < N; i++) {
    S[i] = -1; // does not matter, rate based
  }

  for (size_t i = 0; i < N; i++) {
    previousSpike[i] = -0.1;
    refractory[i] = 0; // no neuron is has fired, so no refraction
    firingsPerNeuron[i] = 0;
  }

  mstime_ = 0;
  t = 0;
  cycle_switcher = 0;
  cur_img = 0;
  exc_spikes = 0;
  inh_spikes = 0;
  input_spikes = 0;
  sleepingCycle = false;
  firings.clear();
}

void Opt1Network::load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) {
  // reset simulation variables
  this->data = dataset;
  this->labels = labels;
  this->reset_values();
}

bool Opt1Network::generateSpike(unsigned value) {
  // Generating spike train from pixel value
  double firing_rate = value / 4000.; // per millisecond
  if (input_intensity > 0) {
    firing_rate = ((0.06375 + (input_intensity*0.032)) * firing_rate) / 0.06375;
  }
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return r <= firing_rate;
}

void Opt1Network::inputSpikes() {
  for (int i = Nn; i < N; ++i) {
    assert(i - Nn >= 0);
    bool spike = generateSpike(this->data[this->cur_img][i - Nn]);
    if (spike) {
      S[i] = 0.005;
    } else {
      S[i] = -1;
    }
  }
}

void Opt1Network::presentData() {
  if (sleepingCycle) {
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= SLEEP_TIME) {
      cur_img++;
      cur_img %= 60000; // keep going through the images
      cycle_switcher = 0;
      sleepingCycle = false;
      exc_spikes = 0;
      inh_spikes = 0;
      input_spikes = 0;
    }
  } else {
    inputSpikes();
    // Check state of the next cycle
    cycle_switcher++;
    if (cycle_switcher >= IMG_TIME) {
      if (exc_spikes < 5) {
        // not enough activation, present the image again with a higher intensity
        // std::cout << " - Not enough spikes, repeating image" << '\n';
        cycle_switcher = 0;
        input_intensity++;
      } else {
        // std::cout << " - Image: " << cur_img << " input: " << input_spikes << " exc: " << exc_spikes << '\n';
        cycle_switcher = 0;
        sleepingCycle = true;
        input_intensity = 0;
      }
    }
  }
}
void Opt1Network::processPreviousSpikes(int i) {
  // Add up any delayed spikes
  if (spikeQueue[mstime_% max_delay][i] > 0) {
    S[i] += spikeQueue[mstime_ % max_delay][i];
    spikeQueue[mstime_ % max_delay][i] = 0;
  }
}
void Opt1Network::handleExcSpikes(int i) {
  if (refractory[i] > 0) {
    refractory[i]--;
    S[i] = v_reset_e;
    return;
  }
  if (S[i] > (v_thresh_e + thetas[i])) {
    if (learning) {
      // postsynaptic spike in neuron i -> update weight of incoming connection
      updateIncomingWeights(i);
      // update theta for homeostasis
      thetas[i] += theta_plus;
    }
    // Propagate spike, is done only once, since exc has one connection
    S[excTargets[i][0]] += excWeights[i][0];
    S[i] = v_reset_e;  // reset potential
    refractory[i] = 50;   // set refractory period
    if (!learning || record_training) {
      // Store spike
      // int c = int(this->labels[this->cur_img]);
      // firings.push_back(std::make_tuple(mstime_, i, c));
      firingsPerNeuron[i]++;
    }
    #pragma omp atomic
    exc_spikes++;       // count this spike for activation
    previousSpike[i] = t; // set timestamp as latest activation
  }
}
void Opt1Network::handleInhSpikes(int i) {
  if (refractory[i] > 0) {
    refractory[i]--;
    S[i] = v_reset_i;
    return;
  }
  if (S[i] > v_thresh_i) {
    for (size_t j = 0; j < Ne-1; j++) {
      S[inhTargets[i-Ne][j]] += inhWeights[i-Ne][j];
      if (S[inhTargets[i-Ne][j]] < 0) {
        S[inhTargets[i-Ne][j]] = 0;
      }
    }
    refractory[i] = 20;
    S[i] = v_reset_i;  // reset potential
    if (!learning || record_training) {
      // Store spike
      // int c = int(this->labels[this->cur_img]);
      // firings.push_back(std::make_tuple(mstime_, i, c));
      firingsPerNeuron[i]++;
    }
    #pragma omp atomic
    inh_spikes++;
    previousSpike[i] = t;
  }
}

void Opt1Network::handleInputSpikes(int i) {
  // Since the input is rate based, use an arbitrary threshold
  if (S[i] > 0) {
    input_spikes++;
    if (learning) {
      connectionTrace[i-Nn] += trace_plus;
    }
    for (size_t j = 0; j < Ne; j++) {
      // presynaptic spike
      spikeQueue[(mstime_ + inputDelays[i-Nn][j]) % max_delay][inputTargets[i-Nn][j]] += inputWeights[i-Nn][j];
    }
    S[i] = -1;  // reset potential
    if (!learning || record_training) {
      // Store spike
      // int c = int(this->labels[this->cur_img]);
      // firings.push_back(std::make_tuple(mstime_, i, c));
      firingsPerNeuron[i]++;
    }
    #pragma omp atomic
    input_spikes++;
    previousSpike[i] = t;
  }
}

void Opt1Network::updateIncomingWeights(int index) {
  for (size_t i = 0; i < Nd; i++) {
    float dv = connectionTrace[i] - stdp_offset;
    float dw = (wmax - (*incomingWeights[index][i])) / wmax;
    float update = stdp_lr_pre * dv * pow(dw, 4);
    (*incomingWeights[index][i]) += update;
    if ( (*incomingWeights[index][i]) > wmax) {
      (*incomingWeights[index][i]) = wmax;
    } else if ( (*incomingWeights[index][i]) < wmin) {
      (*incomingWeights[index][i]) = wmin;
    }
  }
}

void Opt1Network::decayTrace() {
#pragma omp parallel for
  for (size_t i = 0; i < Nd; i++) {
    if (connectionTrace[i] > 0) {
      float diff;
      // Fix rounding errors
      if (t <= previousSpike[i+Nn]+0.000001) {
        diff = 0.0001;
      } else {
        diff = t - previousSpike[i+Nn];
      }
      connectionTrace[i] *= exp(-tau_trace_pre / diff);
    }
  }
}

void Opt1Network::decayNeurons() {
#pragma omp parallel for
  for (size_t i = 0; i < Ne; i++) {
    float diff;
    // Fix rounding errors
    if (t <= previousSpike[i]+0.000001) {
      diff = 0.0001;
    } else {
      diff = t - previousSpike[i];
    }
    S[i] *= exp(-taue / diff);
  }
}
void Opt1Network::decayTheta() {
#pragma omp parallel for
  for (size_t i = 0; i < Ne; i++) {
    float diff;
    // Fix rounding errors
    if (t <= previousSpike[i]+0.000001) {
      diff = 0.0001;
    } else {
      diff = t - previousSpike[i];
    }
    double times = exp(-tau_theta / diff);
    thetas[i] *= times;
    if (thetas[i] < 0) {
      thetas[i] = 0;
    }
  }
}

void Opt1Network::cycle() {
  size_t i = 0;
  // Receive input spikes from image (or don't in an inactive cycle)
  presentData();

  // Add up spikes from the queue if the spike should be applied now
  #pragma omp parallel for
  for (i = 0; i < Ne; i++) {
    processPreviousSpikes(i);
  }

  // Check whether a spike occurs in a neuron, and put that spike in the queue
  // at the given delay
  #pragma omp parallel for
  for (i = 0; i < Ne; i++) {
    handleExcSpikes(i);
  }
  // Check if spikes in the exc layer have made inh neurons spike
  #pragma omp parallel for
  for (i = Ne; i < Nn; i++) {
    handleInhSpikes(i);
  }
  // Lastly, handle the spikes created by input image
  #pragma omp parallel for
  for (i = Nn; i < N; i++) {
    handleInputSpikes(i);
  }

  // Exponential decay on the neuron state
  decayNeurons();
  if (learning) {
    // Exponential decay on the weight traces
    decayTrace();
    // Exponential decay on the theshold thetas
    decayTheta();
  }
}

void Opt1Network::labelNeurons() {
  // For each neuron, count spikes per class
  int classSpikes[N][10];
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < 10; j++) {
      classSpikes[i][j] = 0;
    }
  }

  // Iterate through dataset once
  int last_img = 0;
  while (cur_img < label_limit) {
    if (cur_img != last_img) {
      // When we're done with this image, count the firings that occurred
      // per neuron per class
      for (size_t i = 0; i < N; i++) {
        classSpikes[i][int(labels[last_img])] += firingsPerNeuron[i];
      }
      for (size_t i = 0; i < N; i++) {
        firingsPerNeuron[i] = 0;
      }

      last_img = cur_img;
    }
    cycle();
    t += dt;
    mstime_++;
    if (cur_img % 1000 == 0) {
      std::cout << std::to_string(cur_img) << '\n';
    }
    // std::cout << '\r' << "Progress: " << std::setw(8) << std::setfill(' ')
    //           << (cur_img / float(label_limit))<< std::flush;
  }
  // std::cout << '\n';
  std::cout << "No. of labeling spikes: " << firings.size() << '\n';
  // For each neuron, if its response in this cycle was higher than the
  // previous highest, update the class associated with this neuron

  for (size_t i = 0; i < Ne; i++) {
    // std::cout << "Spike count neuron: " << std::setw(3) << i << ": { ";
    // std::cout << std::setw(3) << classSpikes[i][0] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][1] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][2] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][3] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][4] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][5] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][6] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][7] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][8] << ", ";
    // std::cout << std::setw(3) << classSpikes[i][9] << " } | Highest class: ";
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
    // std::cout << neuronClass[i] << " with: " << highest << " thresh: " << v_thresh_e + thetas[i] << '\n';
  }
}

int Opt1Network::getLabelFromSpikes() {
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
  int last_img = cur_img;
  // store the average intensity of the image s.t. the evaluation loop can read
  // it out
  getImageAvgIntensity();

  while (exc_spikes < 5 || mstime_ < IMG_TIME) {
    cycle();
    t += dt;
    mstime_++;
  }

  for (size_t i = 0; i < Ne; i++) {
    // label associated with this neuron
    int label = neuronClass[i];
    if (label == -1) {
      continue;
    }
    classSpikes[label][0]++;
    classSpikes[label][1] += firingsPerNeuron[i];
  }

  // default answer is 11, such that not coming up with a different answer
  // leads to a 0% accuracy
  float highest = 0.;
  int answer = 11;
  for (size_t i = 0; i < 10; i++) {
    float avg = classSpikes[i][1] / float(classSpikes[i][0]);
    // std::cout << "Class: " << i << " spikes:" << classSpikes[i][1] << " Neurons: " << classSpikes[i][0] << " score: " << avg << '\n';
    if (avg > highest) {
        highest = avg;
        answer = i;
    }
  }
  return answer;
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*                             PLOTTING & OUTPUT                              *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
 void Opt1Network::getImageAvgIntensity() {
   int image_intensity = 0;
   for (size_t i = 0; i < Nd; i++) {
     image_intensity += this->data[this->cur_img][i];
   }
   lastIntensity = image_intensity / float(Nd);
 }

void Opt1Network::plotSpikes() {
  for (auto spike : firings) {
    std::cerr << std::get<0>(spike) << ", " << std::get<1>(spike) << '\n';
  }
}

void Opt1Network::plotWeights() {
  int zero_w = 0;
  for (size_t i = 0; i < Nd; i++) {
    for (size_t j = 0; j < Ne; j++) {
      if (inputWeights[i][j] == wmin) {
        zero_w++;
      }
      std::cerr << inputWeights[i][j] << '\n';
    }
  }
  std::cout << "Number of 0 weights: " << zero_w << '\n';
}

void Opt1Network::saveWeights(std::string filename) {
  std::ofstream weightFile;
  weightFile.open(filename);

  for (size_t j = 0; j < Ne; j++) {
    weightFile << excWeights[j][0] << "," << '\n';
  }
  for (size_t j = 0; j < Ni; j++) {
    for (size_t i = 0; i < Ne-1; i++) {
      weightFile << inhWeights[j][i] << ",";
    }
    weightFile << '\n';
  }
  for (size_t j = 0; j < Nd; j++) {
    for (size_t i = 0; i < Ne; i++) {
      weightFile << inputWeights[j][i] << ",";
    }
    weightFile << '\n';
  }
  weightFile.close();
  std::cout << "---- Saved weights" << '\n';
}

void Opt1Network::loadWeights(std::string filename) {
  std::ifstream weightFile;
  weightFile.open(filename);
  std::string line;
  int index = 0;
  while (std::getline(weightFile,line)) {
    std::stringstream lineStream(line);
    std::string cell;
    int j = 0;
    while(std::getline(lineStream,cell, ',')) {
      if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        // This checks for a trailing comma with no data after it.
        break;
      }
      float w = std::stof(cell);
      if (index < Ne) {
        excWeights[index][j] = w;
      } else if (index >= Ne && index < Nn) {
        inhWeights[index-Ne][j] = w;
      } else {
        inputWeights[index-Nn][j] = w;
      }
      j++;
    }
    index++;
  }
  std::cout << "---- Loaded weights" << '\n';
}

void Opt1Network::saveThetas(std::string filename) {
  std::ofstream thetaFile;
  thetaFile.open(filename);

  for (size_t j = 0; j < Ne; j++) {
    thetaFile << thetas[j] << '\n';
  }
  thetaFile.close();
  std::cout << "---- Saved thetas" << '\n';
}
void Opt1Network::loadThetas(std::string filename) {
  std::ifstream thetaFile;
  thetaFile.open(filename);
  std::string line;
  int index = 0;
  while (std::getline(thetaFile,line)) {
    float w = std::stof(line);
    thetas[index] = w;
    index++;
  }
  std::cout << "---- Loaded thetas" << '\n';
}
void Opt1Network::saveNeuronClasses(std::string filename) {
  std::ofstream neuronClassFile;
  neuronClassFile.open(filename);

  for (size_t j = 0; j < Ne; j++) {
    neuronClassFile << neuronClass[j] << '\n';
  }
  neuronClassFile.close();
  std::cout << "---- Saved neuron classes" << '\n';
}

void Opt1Network::loadNeuronClasses(std::string filename) {
  std::ifstream neuronClassFile;
  neuronClassFile.open(filename);
  std::string line;
  int index = 0;
  while (std::getline(neuronClassFile,line)) {
    int w = std::stoi(line);
    neuronClass[index] = w;
    index++;
  }
  std::cout << "---- Loaded neuron classes" << '\n';
}

void Opt1Network::saveStates() {
  std::ofstream outfile("states.bin", std::ios_base::app);
  if (outfile.is_open()) {
    outfile << "Time: " << t;
    for (size_t i = 0; i < Ne; i++) {
      if (S[i] > 0.013) {
        outfile << " Neuron " << i << " state: " << S[i];
      }
    }
    outfile << "\n";
    outfile.close();
  }
}

void Opt1Network::showWeightExtrema() {
  float highest = 0;
  float lowest = 1;
  size_t k=0,l=0,m=0,n=0;
  for (size_t i = 0; i < Nd; i++) {
    for (size_t j = 0; j < Ne; j++) {
      if (inputWeights[i][j] > highest) {
        highest = inputWeights[i][j];
        k = i;
        l = j;
      } else if (inputWeights[i][j] < lowest) {
        lowest = inputWeights[i][j];
        m = i;
        n = j;
      }
    }
  }
  std::cout << "Highest weight is " << k << " to " << inputTargets[m][n] << " with weight: " << highest << '\n';
  std::cout << "Lowest weight is " << m << " to " << inputTargets[m][n] << " with weight: " << lowest << '\n';
}
void Opt1Network::showThetaExtrema() {
  float highest = 0;
  float lowest = 1;
  size_t k=0,l=0;
  for (size_t i = 0; i < Ne; i++) {
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

void Opt1Network::plotTrace() {
  for (size_t i = 0; i < Nd; i++) {
    std::cerr << mstime_ << ", " << i << ", " << connectionTrace[i] << '\n';
  }
}
void Opt1Network::plotNeurons() {
  for (size_t i = 0; i < Ne; i++) {
    std::cerr << mstime_ << ", " << i << ", " << S[i] << '\n';
  }
}

void Opt1Network::plotWeightImage() {
  float weight2im[Ne][28][28];
  for (size_t i = 0; i < Ne; i++) {
    for (size_t j = 0; j < Nd; j++) {
      for (size_t k = 0; k < Ne; k++) {
        if ( inputTargets[j][k] == i) {
          weight2im[i][(j-Nn) / 28][(j-Nn) % 28] = inputWeights[j][k];
        }
      }
    }
  }
  for (size_t i = 0; i < Ne; i++) {
    for (size_t k = 0; k < 28; k++) {
      for (size_t j = 0; j < 28; j++) {
        std::cerr << weight2im[i][k][j] << ", ";
      }
      std::cerr << '\n';
    }
  }
}

void Opt1Network::plotFiringRates() {
  int spikesPerNeuron[N] = {0};
  for (size_t i = 0; i < firings.size(); i++) {
    // plot images from class 0
    if (std::get<2>(firings[i]) == 8) {
      spikesPerNeuron[std::get<1>(firings[i])]++;
    }
  }

  for (size_t i = 0; i < Nd; i++) {
    if (i % 28 == 0) {
      std::cerr << '\n';
    }
    std::cerr << " " << spikesPerNeuron[i+Nn];
  }
}

void Opt1Network::showNeuronStates() {
  for (size_t i = 0; i < Ne; i++) {
    std::cout << "Neuron: " << i << " state: " << S[i] << " Refrac: " << refractory[i] << " threshold: " << v_thresh_e + thetas[i]<< '\n';
  }
}

void Opt1Network::showTraces() {
  for (size_t i = 0; i < Nd; i++) {
    std::cout << "Neuron: " << i << " trace: " << connectionTrace[i] << '\n';
  }
}


void Opt1Network::liveWeightUpdates() {
  float weight2im[Ne][28][28];
  for (size_t i = 0; i < Ne; i++) {
    for (size_t j = 0; j < Nd; j++) {
      weight2im[i][j % 28][j / 28] = (*incomingWeights[i][j]);
    }
  }
  for (size_t i = 0; i < int(sqrt(Ne)); i++) {
    for (size_t l = 0; l < int(sqrt(Ne)); l++) {
      for (size_t k = 0; k < 28; k++) {
        for (size_t j = 0; j < 28; j++) {
          float x = weight2im[(i*int(sqrt(Ne))) + l][k][j];
          im((i*28) + k,(l*28)+j) = char(ceil((x/wmax)*255));
        }
      }
    }
  }
  cimg_library::CImg<unsigned char> newIm(Network::im);
  Network::dis.display(newIm);
}
