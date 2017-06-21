#ifndef __MATCHER_CUH__
#define __MATCHER_CUH__

#include <vector>

#include "minutia.cuh"

__host__
void matchTemplate(
  const std::vector<Minutia>& minutiae1,
  const std::vector<char>& cylinderValidities1,
  const std::vector<char>& cellValidities1,
  const std::vector<char>& cellValues1,
  const std::vector<Minutia>& minutiae2,
  const std::vector<char>& cylinderValidities2,
  const std::vector<char>& cellValidities2,
  const std::vector<char>& cellValues2,
  std::vector<float>& matrix);

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
  float *devMatrix);

#endif
