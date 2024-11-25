#include "parallel_tree.h"
#include <omp.h>
#include <algorithm> // Para std::max

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {
  // Configuramos el número de hilos para OpenMP
  omp_set_num_threads(4);
}

double ParallelTree::calculateMaxAverage() {
  // Llamamos a la función recursiva paralela
  return calculateMaxAverageInternal(this);
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if (node_ptr == nullptr) {
    return 0.0; // Caso base: si el nodo es nulo, devolvemos 0
  }

  // Suma de datos del nodo actual
  double sum = 0.0;
  int cont = 0;

  for (double value : node_ptr->sensor_data) {
    if (value > 0.0) {
      sum += value;
      cont++;
    }
  }

  double current_avg = (cont > 0) ? sum / static_cast<double>(cont) : 0.0;

  // Variables para almacenar los valores máximos de las ramas
  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // Paralelizamos las ramas izquierda y derecha si existen
  #pragma omp parallel sections
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

  // Devolvemos el máximo entre el nodo actual y sus hijos
  return std::max(current_avg, std::max(max_avg_left, max_avg_right));
}
