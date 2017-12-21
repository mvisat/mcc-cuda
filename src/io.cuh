#ifndef __IO_CUH__
#define __IO_CUH__

#include <vector>

#include "minutia.cuh"

bool loadMinutiaeFromFile(
    const char *input,
    int &width, int &height, int &dpi, int &n,
    std::vector<Minutia> &minutiae);

bool loadTemplateFromFile(
    const char *input,
    int &width, int &height, int &dpi, int &n,
    std::vector<Minutia> &minutiae,
    int &m,
    std::vector<char> &cylinderValidities,
    std::vector<char> &cellValidities,
    std::vector<char> &cellValues);

bool saveTemplateToFile(
    const char *output,
    int width, int height, int dpi, int n,
    const std::vector<Minutia> &minutiae,
    int m,
    const std::vector<char> &cylinderValidities,
    const std::vector<char> &cellValidities,
    const std::vector<char> &cellValues);

bool saveSimilarityToFile(
    const char *output,
    const int n, const int m,
    const std::vector<float> &matrix);

#endif
