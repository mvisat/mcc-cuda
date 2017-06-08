#ifndef __TEMPLATE_CUH__
#define __TEMPLATE_CUH__

#include <vector>

#include "minutia.cuh"

__host__
void buildTemplate(
  const std::vector<Minutia>&,
  const std::vector<char>&,
  int, int,
  std::vector<char>&,
  std::vector<char>&,
  std::vector<char>&);

#endif
