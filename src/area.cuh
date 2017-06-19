#ifndef __AREA_CUH__
#define __AREA_CUH__

#include <vector>

#include "minutia.cuh"

__host__
std::vector<char> buildValidArea(
  const std::vector<Minutia>& minutiae,
  const int width, const int height);

#endif
