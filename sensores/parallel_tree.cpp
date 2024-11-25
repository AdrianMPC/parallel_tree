#include "parallel_tree.h"
#include <omp.h>

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
  if(node_ptr == nullptr) return 0.0;

  // Suma de datos del sensor en el nodo actual (secundario, mantener secuencial)
  double sum = 0.0;
  int cont = 0;
  for(size_t i = 0; i < node_ptr->sensor_data.size(); ++i) {
    if(node_ptr->sensor_data[i] > 0.0) {
      sum += node_ptr->sensor_data[i];
      cont += 1;
    }
  }

  double current_avg = (cont > 0) ? sum / (double)cont : 0.0;

  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // Evitar tareas innecesarias, solo paralelizar ramas si son no triviales
  #pragma omp parallel sections if(node_ptr->left || node_ptr->right)
  {
    #pragma omp section
    {
      if (node_ptr->left != nullptr) {
        max_avg_left = calculateMaxAverageInternal(node_ptr->left);
      }
    }

    #pragma omp section
    {
      if (node_ptr->right != nullptr) {
        max_avg_right = calculateMaxAverageInternal(node_ptr->right);
      }
    }
  }

  // Retornamos el máximo del promedio del nodo y sus hijos
  return std::max(std::max(current_avg, max_avg_left), max_avg_right);
}
