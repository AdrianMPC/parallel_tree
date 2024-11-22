#include "parallel_tree.h"
#include <omp.h>

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {}

double ParallelTree::calculateMaxAverage() {
  return calculateMaxAverageInternal(this);
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if(node_ptr == nullptr) return 0.0;

  // suma los datos del sensor en el nodo actual
  double sum = 0.0;
  int cont = 0;
  #pragma omp parallel for private(value) reduction(+:sum, cont)
  for(int value : node_ptr->sensor_data) {
    if(value > 0.0) {
      sum += value;
      cont += 1;
    }
  }
  // obtnemos promedio
  double current_avg = 0.0;
  if(cont > 0) current_avg = sum / (double)cont;

  // llamadas recursivas para los hijos
  double max_avg_left = calculateMaxAverageInternal(node_ptr->left);
  double max_avg_right = calculateMaxAverageInternal(node_ptr->right);

  // retornamos el máximo del promedio del nodo y sus hijos
  return std::max(std::max(current_avg, max_avg_left), max_avg_right);
}

void ParallelTree::insert(const std::vector<double>& data) {
  insertInternal(this, data);
  contadorEstaciones++;
}

void ParallelTree::insertInternal(SensorTree* node_ptr,
                                    const std::vector<double>& data) {
  if(node_ptr == nullptr) {
    node_ptr = new ParallelTree(data);
    return;
  } else if(node_ptr->left == nullptr) {
    node_ptr->left = new ParallelTree(data);
    return;
  } else if(node_ptr->right == nullptr) {
    node_ptr->right = new ParallelTree(data);
    return;
  }

  if(node_ptr->left != nullptr) insertInternal(node_ptr->left, data);
  if(node_ptr->right != nullptr) insertInternal(node_ptr->right, data);
}
