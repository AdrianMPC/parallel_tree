#include "parallel_tree.h"
#include <omp.h>
#include <algorithm>
#include <vector>

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {
  // Configurar un número razonable de hilos
  omp_set_num_threads(4);
}

double ParallelTree::calculateMaxAverage() {
  double maxAverage = 0.0;

  // Región paralela con una sola tarea inicial
  #pragma omp parallel
  {
    #pragma omp single
    {
      maxAverage = calculateMaxAverageInternal(this);
    }
  }

  return maxAverage;
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if (node_ptr == nullptr) return 0.0;

  // Calcular el promedio del nodo actual
  double sum = 0.0;
  int count = 0;

  for (double value : node_ptr->sensor_data) {
    if (value > 0.0) {
      sum += value;
      count++;
    }
  }

  double currentAverage = (count > 0) ? sum / count : 0.0;

  double maxLeft = 0.0;
  double maxRight = 0.0;

  // Crear tareas para las ramas izquierda y derecha
  #pragma omp task shared(maxLeft) if (node_ptr->left != nullptr)
  {
    maxLeft = calculateMaxAverageInternal(node_ptr->left);
  }

  #pragma omp task shared(maxRight) if (node_ptr->right != nullptr)
  {
    maxRight = calculateMaxAverageInternal(node_ptr->right);
  }

  // Esperar a que todas las tareas finalicen
  #pragma omp taskwait

  // Retornar el máximo entre el promedio actual y los máximos de las ramas
  return std::max({currentAverage, maxLeft, maxRight});
}
