#include "matcher.cuh"
#include "constants.cuh"
#include "errors.h"

using namespace std;

__global__
void binarizedTemplate(
    char *cellValidities,
    char *cellValues,
    int *binarizedValidities,
    int *binarizedValues) {
  int idxMinutia = blockIdx.x;
  int idxInt = threadIdx.x;
  int bits = sizeof(int) * 8;
  int intPerCylinder = NC / bits;
  int idx = idxMinutia * intPerCylinder + idxInt;
  int idxBit = idxMinutia * NC + idxInt * bits;

  int validity = 0, value = 0;
  for (int i = 0; i < bits; ++i) {
    validity |= cellValidities[idxBit+i] << i;
    value |= cellValues[idxBit+i] << i;
  }
  binarizedValidities[idx] = validity;
  binarizedValues[idx] = value;
}

__host__
float matchTemplate(
    const vector<char>& cylinderValidities1,
    const vector<char>& cellValidities1,
    const vector<char>& cellValues1,
    const vector<char>& cylinderValidities2,
    const vector<char>& cellValidities2,
    const vector<char>& cellValues2) {
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

  int bits = sizeof(int) * 8;
  int *devBinarizedValidities1, *devBinarizedValues1;
  int *devBinarizedValidities2, *devBinarizedValues2;
  size_t devBinarizedValidities1Size = (cellValidities1.size() / bits) * sizeof(int);
  size_t devBinarizedValidities2Size = (cellValidities2.size() / bits) * sizeof(int);
  size_t devBinarizedValues1Size = (cellValues1.size() / bits) * sizeof(int);
  size_t devBinarizedValues2Size = (cellValues2.size() / bits) * sizeof(int);
  handleError(
    cudaMalloc(&devBinarizedValidities1, devBinarizedValidities1Size));
  handleError(
    cudaMalloc(&devBinarizedValidities2, devBinarizedValidities2Size));
  handleError(
    cudaMalloc(&devBinarizedValues1, devBinarizedValues1Size));
  handleError(
    cudaMalloc(&devBinarizedValues2, devBinarizedValues2Size));

  binarizedTemplate<<<cylinderValidities1.size(), NC/bits>>>(
    devCellValidities1, devCellValues1, devBinarizedValidities1, devBinarizedValues1);
  binarizedTemplate<<<cylinderValidities2.size(), NC/bits>>>(
    devCellValidities2, devCellValues2, devBinarizedValidities2, devBinarizedValues2);

  // TODO

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
}
