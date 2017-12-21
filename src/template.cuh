#ifndef __TEMPLATE_CUH__
#define __TEMPLATE_CUH__

#include <vector>

#include "minutia.cuh"

__host__
void buildTemplate(
  const std::vector<Minutia>&,
  const int, const int,
  std::vector<char>&,
  std::vector<char>&,
  std::vector<char>&);

__host__
void devBuildTemplate(
  Minutia *devMinutiae, const int n,
  char *devArea, const int width, const int height,
  char *devCylinderValidities,
  char *devCellValidities,
  char *devCellValues);

#endif
