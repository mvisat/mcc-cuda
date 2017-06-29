#include <algorithm>
#include <functional>

#include "consolidation.cuh"
#include "constants.cuh"
#include "util.cuh"
#include "sort.cuh"

using namespace std;

__host__ __device__ __inline__
int getNP(const int rows, const int cols) {
  return MIN_NP +
    roundf(sigmoid(min(rows, cols), TAU_P, MU_P) * (MAX_NP - MIN_NP));
}

__host__
float LSS(const vector<float>& _matrix, const int rows, const int cols) {
  auto matrix(_matrix);
  int n = getNP(rows, cols);
  nth_element(matrix.begin(), matrix.begin()+n, matrix.end(), greater<float>());
  auto sum = accumulate(matrix.begin(), matrix.begin()+n, 0.0f);
  return sum / n;
}

__host__
float devLSS(float *devMatrix, const int rows, const int cols) {
  int n = getNP(rows, cols);
  devBitonicSort(devMatrix, rows*cols);
  vector<float> matrix(n);
  cudaMemcpy(matrix.data(), devMatrix, n * sizeof(float), cudaMemcpyDeviceToHost);
  auto sum = accumulate(matrix.begin(), matrix.end(), 0.0f);
  return sum / n;
}
