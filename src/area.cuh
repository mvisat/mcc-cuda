#ifndef __AREA_CUH__
#define __AREA_CUH__

#include <vector>

#include "minutia.cuh"

__host__
std::vector<char> buildValidArea(std::vector<Minutia>& minutiae, int width, int height);

__host__
std::vector<Minutia> buildConvexHull(std::vector<Minutia>& minutiae);

#endif
