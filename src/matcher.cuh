#ifndef __MATCHER_CUH__
#define __MATCHER_CUH__

#include <vector>

#include "minutia.cuh"

__host__
float matchTemplate(
  const std::vector<Minutia>& minutiae1,
  const std::vector<char>& cylinderValidities1,
  const std::vector<char>& cellValidities1,
  const std::vector<char>& cellValues1,
  const std::vector<Minutia>& minutiae2,
  const std::vector<char>& cylinderValidities2,
  const std::vector<char>& cellValidities2,
  const std::vector<char>& cellValues2,
  std::vector<float>& matrix);

#endif
