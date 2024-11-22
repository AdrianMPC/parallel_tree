#ifndef PARALLEL_H
#define PARALLEL_H

#include "sensor_tree.h"

class ParallelTree : public SensorTree {
 public:
  int contadorEstaciones;
  ParallelTree(const std::vector<double>&);
  double calculateMaxAverageInternal(SensorTree* node_ptr) override;
  double calculateMaxAverage() override;
  void insertInternal(SensorTree*, const std::vector<double>&) override;
  void insert(const std::vector<double>&) override;
};

#endif  // PARALLEL_H
