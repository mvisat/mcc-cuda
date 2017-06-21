#include <algorithm>
#include <functional>

#include "consolidation.cuh"
#include "constants.cuh"
#include "util.cuh"

using namespace std;

__host__
float LSS(const vector<float>& _matrix, const int rows, const int cols) {
  auto matrix(_matrix);
  auto sigmoid = [&](int value, float tau, float mu) {
    return 1.0f / (1.0f + expf(-tau * (value-mu)));
  };
  int n = MIN_NP + roundf(sigmoid(min(rows, cols), TAU_P, MU_P) * (MAX_NP - MIN_NP));
  nth_element(matrix.begin(), matrix.begin()+n, matrix.end(), greater<float>());
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += matrix[i];
  return sum / n;
}
