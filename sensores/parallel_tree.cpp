#include "parallel_tree.h"
#include <omp.h>

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {
  // Limitar el número de hilos para evitar sobrecarga de memoria
  omp_set_num_threads(4);
}

double ParallelTree::calculateMaxAverage() {
  double result = 0.0;

  // Crear una única región paralela para calcular desde la raíz
  #pragma omp parallel
  {
    // Solo un hilo crea la tarea inicial de calcular el promedio máximo
    #pragma omp single
    result = calculateMaxAverageInternal(this);
  }

  return result;
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if (node_ptr == nullptr) return 0.0;

  // Sumar datos del sensor en el nodo actual
  double sum = 0.0;
  int cont = 0;

  // No paralelizamos esta parte ya que suele ser rápida
  for (size_t i = 0; i < node_ptr->sensor_data.size(); ++i) {
    if (node_ptr->sensor_data[i] > 0.0) {
      sum += node_ptr->sensor_data[i];
      cont += 1;
    }
  }

  // Calcular promedio en el nodo actual
  double current_avg = (cont > 0) ? sum / (double)cont : 0.0;

  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // Utilizar secciones paralelas para recorrer ramas del árbol
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

  // Retornar el máximo promedio entre el nodo actual y sus hijos
  return std::max(current_avg, std::max(max_avg_left, max_avg_right));
}

void ParallelTree::insert(const std::vector<double>& data) {
  insertInternal(this, data);
  contadorEstaciones++;
}

void ParallelTree::insertInternal(SensorTree* node_ptr, const std::vector<double>& data) {
  if (node_ptr == nullptr) {
    node_ptr = new ParallelTree(data);
    return;
  } else if (node_ptr->left == nullptr) {
    node_ptr->left = new ParallelTree(data);
    return;
  } else if (node_ptr->right == nullptr) {
    node_ptr->right = new ParallelTree(data);
    return;
  }

  // Inserción recursiva en los subárboles
  if (node_ptr->left != nullptr) insertInternal(node_ptr->left, data);
  if (node_ptr->right != nullptr) insertInternal(node_ptr->right, data);
}
