#ifndef BASE_H
#define BASE_H

#include <vector>

class SensorTree {
 public:
  // datos de los sensores en esta sub-estación
  std::vector<double> sensor_data;
  SensorTree* left;
  SensorTree* right;

  SensorTree(const std::vector<double>&);
  virtual double calculateMaxAverageInternal(SensorTree*) = 0;
  virtual double calculateMaxAverage() = 0;
  virtual void insertInternal(SensorTree*, const std::vector<double>&) = 0;
  virtual void insert(const std::vector<double>&) = 0;
  ~SensorTree();
};

#endif  // BASE_H
