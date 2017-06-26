#pragma once

#ifdef Success
#undef Success
#endif

#include <vector>

class Network {
public:
  Network(int train, int label, int test, bool learn, bool record) :
    train_limit(train),
    label_limit(label),
    test_limit(test),
    learning(learn),
    record_training(record)
  {}

  virtual void initialize_params() = 0;
  virtual void cycle() = 0;
  virtual void plotSpikes() = 0;
  virtual void showWeightExtrema() = 0;
  virtual void showThetaExtrema() = 0;
  virtual void load_dataset(std::vector<std::vector<unsigned char, std::allocator<unsigned char>>>& dataset, std::vector<unsigned char>& labels) = 0;
  virtual void reset_values() = 0;
  virtual void labelNeurons() = 0;
  virtual int getLabelFromSpikes() = 0;
  virtual void saveWeights(std::string filename) = 0;
  virtual void loadWeights(std::string filename) = 0;
  virtual void saveThetas(std::string filename) = 0;
  virtual void loadThetas(std::string filename) = 0;
  virtual void saveNeuronClasses(std::string filename) = 0;
  virtual void loadNeuronClasses(std::string filename) = 0;
  virtual void getImageAvgIntensity() = 0;

  bool record_training = false;
  bool learning = true;
  int train_limit = 100;
  int label_limit = 100;
  int test_limit  = 100;
  double t = 0;               // time counter as in Brian(python framework)
  double dt = 0.0001;         // time addition on every step
  unsigned mstime_ = 0;       // current time stamp in natural numbers
  unsigned cur_img = 0;       // index of the current image
  float lastIntensity = 0;    // average intensity of the current image
  unsigned exc_spikes = 0;  // number of spikes in the exc layer during the presentation of the current image
  unsigned inh_spikes = 0;  // number of spikes in the inh layer during the presentation of the current image
  unsigned input_spikes = 0;  // number of spikes in the input layer during the presentation of the current image


  // Dataset used as input
  std::vector<std::vector<unsigned char, std::allocator<unsigned char>>> data;
  // Labels corresponding to the input
  std::vector<unsigned char> labels;
  // The timestamp, index and label of spikes stored
  std::vector<std::tuple<float, int, int>> firings;
};
