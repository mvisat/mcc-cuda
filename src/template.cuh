#ifndef __TEMPLATE_CUH__
#define __TEMPLATE_CUH__

#include <vector>

#include "minutia.cuh"

__host__
std::vector<char2> buildTemplate(
  const std::vector<Minutia>&,
  const std::vector<char>&,
  int rows, int cols);

#endif
