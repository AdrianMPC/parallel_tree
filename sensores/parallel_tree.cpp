#include "parallel_tree.h"
#include <omp.h>
#include <algorithm>

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {
  // Usar un número de hilos controlado
  omp_set_num_threads(4);
}

double ParallelTree::calculateMaxAverage() {
  double result = 0.0;

  // Mantener una única región paralela desde la raíz
  #pragma omp parallel
  {
    #pragma omp single nowait
    result = calculateMaxAverageInternal(this);
  }

  return result;
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if (node_ptr == nullptr) return 0.0;

  // Suma de datos del sensor en el nodo actual
  double sum = 0.0;
  int count = 0;
  for (size_t i = 0; i < node_ptr->sensor_data.size(); ++i) {
    if (node_ptr->sensor_data[i] > 0.0) {
      sum += node_ptr->sensor_data[i];
      count += 1;
    }
  }

  double current_avg = (count > 0) ? sum / (double)count : 0.0;

  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // Paralelizar ramas no triviales
  #pragma omp task shared(max_avg_left) if (node_ptr->left != nullptr)
  {
    max_avg_left = calculateMaxAverageInternal(node_ptr->left);
  }

  #pragma omp task shared(max_avg_right) if (node_ptr->right != nullptr)
  {
    max_avg_right = calculateMaxAverageInternal(node_ptr->right);
  }

  // Esperar a que todas las tareas terminen
  #pragma omp taskwait

  // Retornar el máximo del promedio del nodo actual y sus hijos
  return std::max({current_avg, max_avg_left, max_avg_right});
}
