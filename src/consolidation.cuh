#ifndef __CONSOLIDATION_CUH__
#define __CONSOLIDATION_CUH__

#include <vector>

#include "minutia.cuh"

__host__
float LSS(const std::vector<float>& _matrix, const int rows, const int cols);

__host__
float devLSS(float *devMatrix, const int rows, const int cols);

__host__
float LSSR(const std::vector<float> &_gamma,
    const int rows, const int cols,
    const std::vector<Minutia> &minutiaeA, const std::vector<Minutia> &minutiaeB);

#endif
