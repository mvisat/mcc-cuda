#include "binarization.cuh"
#include "constants.cuh"
#include "errors.h"

__global__
void binarized(
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

__host__
void devBinarizedTemplate(
    const int n,
    char *devCellValidities,
    char *devCellValues,
    unsigned int *devBinarizedValidities,
    unsigned int *devBinarizedValues) {
  int intPerCylinder = NC/BITS;
  binarized<<<n, intPerCylinder>>>(
    devCellValidities, devCellValues, devBinarizedValidities, devBinarizedValues);
  handleError(
    cudaPeekAtLastError());
}
