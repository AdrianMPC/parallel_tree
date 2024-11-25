#include "parallel_tree.h"
#include <omp.h>

ParallelTree::ParallelTree(const std::vector<double>& data)
    : SensorTree(data), contadorEstaciones(1) {}

double ParallelTree::calculateMaxAverage() {
  return calculateMaxAverageInternal(this);
}

double ParallelTree::calculateMaxAverageInternal(SensorTree* node_ptr) {
  if(node_ptr == nullptr) return 0.0;

  // Suma los datos del sensor en el nodo actual
  double sum = 0.0;
  int cont = 0;

  // Comprueba si hay suficientes datos para paralelizar
  if (node_ptr->sensor_data.size() > 1000) {
    // Paraleliza el cálculo de la suma si hay suficientes datos
    #pragma omp parallel for reduction(+:sum, cont)
    for(size_t i = 0; i < node_ptr->sensor_data.size(); i++) {
      double value = node_ptr->sensor_data[i];
      if(value > 0.0) {
        sum += value;
        cont += 1;
      }
    }
  } else {
    // Calcula la suma secuencialmente si el tamaño de los datos es pequeño
    for(double value : node_ptr->sensor_data) {
      if(value > 0.0) {
        sum += value;
        cont += 1;
      }
    }
  }

  // Obtenemos promedio
  double current_avg = 0.0;
  if(cont > 0) current_avg = sum / (double)cont;

  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // Paraleliza la recursión usando tareas
  #pragma omp parallel
  {
    #pragma omp single
    {
      if (node_ptr->left != nullptr) {
        #pragma omp task
        max_avg_left = calculateMaxAverageInternal(node_ptr->left);
      }

      if (node_ptr->right != nullptr) {
        #pragma omp task
        max_avg_right = calculateMaxAverageInternal(node_ptr->right);
      }

      #pragma omp taskwait
    }
  }

  // Retornamos el máximo del promedio del nodo y sus hijos
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
