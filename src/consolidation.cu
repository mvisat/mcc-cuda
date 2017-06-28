#include <algorithm>
#include <functional>

#include "consolidation.cuh"
#include "constants.cuh"
#include "util.cuh"

using namespace std;

__host__
float LSS(const vector<float>& _matrix, const int rows, const int cols) {
  auto matrix(_matrix);
  int n = MIN_NP + roundf(sigmoid(min(rows, cols), TAU_P, MU_P) * (MAX_NP - MIN_NP));
  nth_element(matrix.begin(), matrix.begin()+n, matrix.end(), greater<float>());
  auto sum = accumulate(matrix.begin(), matrix.begin()+n, 0.0f);
  return sum / n;
}
