#include "matcher.cuh"
#include "binarization.cuh"
#include "minutia.cuh"
#include "constants.cuh"
#include "util.cuh"
#include "errors.h"
#include "debug.h"

using namespace std;

__global__
void computeSimilarity(
    Minutia *minutiae1,
    char *cylinderValidities1,
    unsigned int *binarizedValidities1,
    unsigned int *binarizedValues1,
    Minutia *minutiae2,
    char *cylinderValidities2,
    unsigned int *binarizedValidities2,
    unsigned int *binarizedValues2,
    float *matrix, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) return;

  if (!cylinderValidities1[row] || !cylinderValidities2[col] ||
      floatGreater(
        fabsf(angle(minutiae1[row].theta, minutiae2[col].theta)),
        DELTA_THETA)) {
    matrix[row*cols + col] = 0.0f;
    return;
  }

  int intPerCylinder = NC/BITS;
  int rowIdx = row * intPerCylinder;
  int colIdx = col * intPerCylinder;

  int maskBits = 0, rowBits = 0, colBits = 0, xorBits = 0;
  for (int i = 0; i < intPerCylinder; ++i) {
    auto mask = binarizedValidities1[rowIdx+i] & binarizedValidities2[colIdx+i];
    auto rowValue = binarizedValues1[rowIdx+i] & mask;
    auto colValue = binarizedValues2[colIdx+i] & mask;
    auto xorValue = rowValue ^ colValue;
    maskBits += __popc(mask);
    rowBits += __popc(rowValue);
    colBits += __popc(colValue);
    xorBits += __popc(xorValue);
  }

  bool matchable = maskBits >= MIN_ME_CELLS && (rowBits || colBits);
  float similarity = matchable
    ? (1 - sqrtf(xorBits) / (sqrtf(rowBits)+sqrtf(colBits)))
    : 0.0f;
  matrix[row*cols + col] = similarity;
}

__host__
void devMatchTemplate(
    Minutia *devMinutiae1, const int n,
    char *devCylinderValidities1,
    unsigned int *devBinarizedValidities1,
    unsigned int *devBinarizedValues1,
    Minutia *devMinutiae2, const int m,
    char *devCylinderValidities2,
    unsigned int *devBinarizedValidities2,
    unsigned int *devBinarizedValues2,
    float *devMatrix) {
  int threadPerDim = 32;
  dim3 blockCount(ceilMod(m, threadPerDim), ceilMod(n, threadPerDim));
  dim3 threadCount(threadPerDim, threadPerDim);
  computeSimilarity<<<blockCount, threadCount>>>(
    devMinutiae1, devCylinderValidities1, devBinarizedValidities1, devBinarizedValues1,
    devMinutiae2, devCylinderValidities2, devBinarizedValidities2, devBinarizedValues2,
    devMatrix, n, m);
  handleError(
    cudaPeekAtLastError());
}

__host__
void matchTemplate(
    const vector<Minutia>& minutiae1,
    const vector<char>& cylinderValidities1,
    const vector<char>& cellValidities1,
    const vector<char>& cellValues1,
    const vector<Minutia>& minutiae2,
    const vector<char>& cylinderValidities2,
    const vector<char>& cellValidities2,
    const vector<char>& cellValues2,
    vector<float>& matrix) {

  Minutia *devMinutiae1, *devMinutiae2;
  char *devCylinderValidities1, *devCylinderValidities2;
  char *devCellValidities1, *devCellValidities2;
  char *devCellValues1, *devCellValues2;
  size_t devMinutiae1Size = minutiae1.size() * sizeof(Minutia);
  size_t devMinutiae2Size = minutiae2.size() * sizeof(Minutia);
  size_t devCylinderValidities1Size = cylinderValidities1.size() * sizeof(char);
  size_t devCylinderValidities2Size = cylinderValidities2.size() * sizeof(char);
  size_t devCellValidities1Size = cellValidities1.size()  * sizeof(char);
  size_t devCellValidities2Size = cellValidities2.size()  * sizeof(char);
  size_t devCellValues1Size = cellValues1.size() * sizeof(char);
  size_t devCellValues2Size = cellValues2.size() * sizeof(char);
  handleError(
    cudaMalloc(&devMinutiae1, devMinutiae1Size));
  handleError(
    cudaMemcpy(devMinutiae1, minutiae1.data(), devMinutiae1Size, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devMinutiae2, devMinutiae2Size));
  handleError(
    cudaMemcpy(devMinutiae2, minutiae2.data(), devMinutiae2Size, cudaMemcpyHostToDevice));
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

  int intPerCylinder = NC/BITS;
  unsigned int *devBinarizedValidities1, *devBinarizedValues1;
  unsigned int *devBinarizedValidities2, *devBinarizedValues2;
  size_t devBinarizedValidities1Size = minutiae1.size() * intPerCylinder * sizeof(unsigned int);
  size_t devBinarizedValues1Size = minutiae1.size() * intPerCylinder * sizeof(unsigned int);
  size_t devBinarizedValidities2Size = minutiae2.size() * intPerCylinder * sizeof(unsigned int);
  size_t devBinarizedValues2Size = minutiae2.size() * intPerCylinder * sizeof(unsigned int);
  handleError(
    cudaMalloc(&devBinarizedValidities1, devBinarizedValidities1Size));
  handleError(
    cudaMalloc(&devBinarizedValidities2, devBinarizedValidities2Size));
  handleError(
    cudaMalloc(&devBinarizedValues1, devBinarizedValues1Size));
  handleError(
    cudaMalloc(&devBinarizedValues2, devBinarizedValues2Size));

  devBinarizedTemplate(minutiae1.size(),
    devCellValidities1, devCellValues1,
    devBinarizedValidities1, devBinarizedValues1);
  devBinarizedTemplate(minutiae2.size(),
    devCellValidities2, devCellValues2,
    devBinarizedValidities2, devBinarizedValues2);

  float *devMatrix;
  size_t devMatrixSize = minutiae1.size() * minutiae2.size() * sizeof(float);
  handleError(
    cudaMalloc(&devMatrix, devMatrixSize));

  devMatchTemplate(
    devMinutiae1, minutiae1.size(),
    devCylinderValidities1, devBinarizedValidities1, devBinarizedValues1,
    devMinutiae2, minutiae2.size(),
    devCylinderValidities2, devBinarizedValidities2, devBinarizedValues2,
    devMatrix);

  matrix.resize(minutiae1.size() * minutiae2.size());
  handleError(
    cudaMemcpy(matrix.data(), devMatrix, devMatrixSize, cudaMemcpyDeviceToHost));

  cudaFree(devMinutiae1);
  cudaFree(devMinutiae2);
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
}
