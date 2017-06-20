#include "matcher.cuh"
#include "constants.cuh"
#include "errors.h"
#include "debug.h"

#include <vector>
#include <algorithm>
#include <functional>

#define ceilMod(x,y) (x+y-1)/y

using namespace std;

__global__
void binarizedTemplate(
    char *cellValidities,
    char *cellValues,
    unsigned int *binarizedValidities,
    unsigned int *binarizedValues) {
  int idxMinutia = blockIdx.x;
  int idxInt = threadIdx.x;
  int intPerCylinder = NC / BITS;
  int idx = idxMinutia * intPerCylinder + idxInt;
  int idxBit = idxMinutia * NC + idxInt * BITS;

  unsigned int validity = 0, value = 0;
  for (int i = 0; i < BITS; ++i) {
    validity <<= 1U;
    validity |= cellValidities[idxBit+i];
    value <<= 1U;
    value |= cellValues[idxBit+i];
  }
  binarizedValidities[idx] = validity;
  binarizedValues[idx] = value;
}

__global__
void computeSimilarity(
    unsigned int *binarizedValidities1,
    unsigned int *binarizedValues1,
    unsigned int *binarizedValidities2,
    unsigned int *binarizedValues2,
    float *matrix, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) return;

  int intPerCylinder = NC/BITS;
  int rowIdx = row * intPerCylinder;
  int colIdx = col * intPerCylinder;

  unsigned int matchable = 0;
  int rowBits = 0, colBits = 0, xorBits = 0;

  for (int i = 0; i < intPerCylinder; ++i) {
    auto mask = binarizedValidities1[rowIdx+i] & binarizedValidities2[colIdx+i];
    auto rowValue = binarizedValues1[rowIdx+i] & mask;
    auto colValue = binarizedValues2[colIdx+i] & mask;
    auto xorValue = rowValue ^ colValue;
    matchable += mask;
    rowBits += __popc(rowValue);
    colBits += __popc(colValue);
    xorBits += __popc(xorValue);
  }

  float similarity = matchable && (rowBits+colBits)
    ? (1.0f - sqrtf(xorBits) / (sqrtf(rowBits)+sqrtf(colBits)))
    : 0.0f;
  matrix[row*cols + col] = similarity;
}

__host__
float LSS(const vector<float>& _matrix, int rows, int cols) {
  auto matrix(_matrix);
  auto sigmoid = [&](int value, float tau, float mu) {
    return 1.0f / (1.0f + expf(-tau * (value-mu)));
  };
  int n = MIN_NP + roundf(sigmoid(min(rows, cols), TAU_P, MU_P) * (MAX_NP - MIN_NP));
  debug("NP: %d\n", n);
  nth_element(matrix.begin(), matrix.begin()+n, matrix.end(), greater<float>());
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += matrix[i];
  return sum / n;
}

__host__
float matchTemplate(
    const vector<char>& cylinderValidities1,
    const vector<char>& cellValidities1,
    const vector<char>& cellValues1,
    const vector<char>& cylinderValidities2,
    const vector<char>& cellValidities2,
    const vector<char>& cellValues2,
    vector<float>& matrix) {

  int rows = cylinderValidities1.size();
  int cols = cylinderValidities2.size();

  char *devCylinderValidities1, *devCylinderValidities2;
  char *devCellValidities1, *devCellValidities2;
  char *devCellValues1, *devCellValues2;
  size_t devCylinderValidities1Size = cylinderValidities1.size() * sizeof(char);
  size_t devCylinderValidities2Size = cylinderValidities2.size() * sizeof(char);
  size_t devCellValidities1Size = cellValidities1.size()  * sizeof(char);
  size_t devCellValidities2Size = cellValidities2.size()  * sizeof(char);
  size_t devCellValues1Size = cellValues1.size() * sizeof(char);
  size_t devCellValues2Size = cellValues2.size() * sizeof(char);
  handleError(
    cudaMalloc(&devCylinderValidities1, devCylinderValidities1Size));
  handleError(
    cudaMemcpy(devCylinderValidities1, cylinderValidities1.data(), devCylinderValidities1Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCylinderValidities2, devCylinderValidities2Size));
  handleError(
    cudaMemcpy(devCylinderValidities2, cylinderValidities2.data(), devCylinderValidities2Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValidities1, devCellValidities1Size));
  handleError(
    cudaMemcpy(devCellValidities1, cellValidities1.data(), devCellValidities1Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValidities2, devCellValidities2Size));
  handleError(
    cudaMemcpy(devCellValidities2, cellValidities2.data(), devCellValidities2Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValues1, devCellValues1Size));
  handleError(
    cudaMemcpy(devCellValues1, cellValues1.data(), devCellValues1Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValues2, devCellValues2Size));
  handleError(
    cudaMemcpy(devCellValues2, cellValues2.data(), devCellValues2Size, cudaMemcpyHostToDevice));

  unsigned int *devBinarizedValidities1, *devBinarizedValues1;
  unsigned int *devBinarizedValidities2, *devBinarizedValues2;
  size_t devBinarizedValidities1Size = (cellValidities1.size()/BITS) * sizeof(unsigned int);
  size_t devBinarizedValidities2Size = (cellValidities2.size()/BITS) * sizeof(unsigned int);
  size_t devBinarizedValues1Size = (cellValues1.size()/BITS) * sizeof(unsigned int);
  size_t devBinarizedValues2Size = (cellValues2.size()/BITS) * sizeof(unsigned int);
  handleError(
    cudaMalloc(&devBinarizedValidities1, devBinarizedValidities1Size));
  handleError(
    cudaMalloc(&devBinarizedValidities2, devBinarizedValidities2Size));
  handleError(
    cudaMalloc(&devBinarizedValues1, devBinarizedValues1Size));
  handleError(
    cudaMalloc(&devBinarizedValues2, devBinarizedValues2Size));

  binarizedTemplate<<<cylinderValidities1.size(), NC/BITS>>>(
    devCellValidities1, devCellValues1, devBinarizedValidities1, devBinarizedValues1);
  handleError(
    cudaPeekAtLastError());
  binarizedTemplate<<<cylinderValidities2.size(), NC/BITS>>>(
    devCellValidities2, devCellValues2, devBinarizedValidities2, devBinarizedValues2);
  handleError(
    cudaPeekAtLastError());

  float *devMatrix;
  size_t devMatrixSize = rows * cols * sizeof(float);
  handleError(
    cudaMalloc(&devMatrix, devMatrixSize));

  int threadPerDim = 32;
  dim3 blockCount(ceilMod(rows, threadPerDim), ceilMod(cols, threadPerDim));
  dim3 threadCount(threadPerDim, threadPerDim);
  computeSimilarity<<<blockCount, threadCount>>>(
    devBinarizedValidities1, devBinarizedValues1,
    devBinarizedValidities2, devBinarizedValues2,
    devMatrix, rows, cols);
  handleError(
    cudaPeekAtLastError());

  matrix.resize(rows*cols);
  handleError(
    cudaMemcpy(matrix.data(), devMatrix, devMatrixSize, cudaMemcpyDeviceToHost));

  cudaFree(devCylinderValidities1);
  cudaFree(devCylinderValidities2);
  cudaFree(devCellValidities1);
  cudaFree(devCellValidities2);
  cudaFree(devCellValues1);
  cudaFree(devCellValues2);
  cudaFree(devBinarizedValidities1);
  cudaFree(devBinarizedValidities2);
  cudaFree(devBinarizedValues1);
  cudaFree(devBinarizedValues2);
  cudaFree(devMatrix);

  return LSS(matrix, rows, cols);
}
