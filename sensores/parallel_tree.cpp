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
  int id_thread = omp_get_thread_num();
  
  //#pragma omp parallel for reduction(+:sum, cont)
  for(int value : node_ptr->sensor_data) {
    if(value > 0.0) {
      sum += value;
      cont += 1;
    }
  }
  // obtnemos promedio
  double current_avg = 0.0;
  if(cont > 0) current_avg = sum / (double)cont;

  double max_avg_left = 0.0;
  double max_avg_right = 0.0;

  // llamadas recursivas para los hijos
  if (id_thread == 0)
  {
    #pragma omp parallel
    {
    // :v
      #pragma omp task shared(max_avg_left)
      max_avg_left = calculateMaxAverageInternal(node_ptr->left);
      
      #pragma omp task shared(max_avg_right)
      max_avg_right = calculateMaxAverageInternal(node_ptr->right);

      #pragma omp taskwait
    }
  } else {
    max_avg_left = calculateMaxAverageInternal(node_ptr->left);
    max_avg_right = calculateMaxAverageInternal(node_ptr->right);
  }
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
