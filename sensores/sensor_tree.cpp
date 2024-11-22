#include "sensor_tree.h"

SensorTree::SensorTree(const std::vector<double>& data)
    : sensor_data(data), left(nullptr), right(nullptr) {}

SensorTree::~SensorTree() {
  delete left;
  left = nullptr;
  delete right;
  right = nullptr;
}