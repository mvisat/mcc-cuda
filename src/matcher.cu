#include "matcher.cu"
#include "constants.cuh"

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
  int idxBit = idxMinutia * intPerCylinder + idxInt * bits;

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
    cudaMemcpy(devCylinderValidities1, cylinderValidities1.data(), cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCylinderValidities2, devCylinderValidities2Size));
  handleError(
    cudaMemcpy(devCylinderValidities2, cylinderValidities2.data(), cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValidities1, devCellValidities1Size));
  handleError(
    cudaMemcpy(devCellValidities1, cellValidities1.data(), cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValidities2, devCellValidities2Size));
  handleError(
    cudaMemcpy(devCellValidities2, cellValidities2.data(), cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValues1, devCellValues1Size));
  handleError(
    cudaMemcpy(devCellValues1, cellValues1.data(), cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCellValues2, devCellValues2Size));
  handleError(
    cudaMemcpy(devCellValues2, cellValues2.data(), cudaMemcpyHostToDevice));

  // TODO

  cudaFree(devCylinderValidities1);
  cudaFree(devCylinderValidities2);
  cudaFree(devCellValidities1);
  cudaFree(devCellValidities2);
  cudaFree(devCellValues1);
  cudaFree(devCellValues2);
}
