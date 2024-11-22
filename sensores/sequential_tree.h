#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "sensor_tree.h"

class SequentialTree : public SensorTree {
 public:
  int contadorEstaciones;
  SequentialTree(const std::vector<double>&);
  double calculateMaxAverageInternal(SensorTree* node_ptr) override;
  double calculateMaxAverage() override;
  void insertInternal(SensorTree*, const std::vector<double>&) override;
  void insert(const std::vector<double>&) override;
};

#endif  // SEQUENTIAL_H
