#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>

#include "errors.h"
#include "debug.h"
#include "constants.cuh"
#include "area.cuh"
#include "template.cuh"
#include "matcher.cuh"

using namespace std;

int main() {
  int width, height, dpi, n;
  ifstream stream("data/1_1.txt");
  stream >> width >> height >> dpi >> n;
  vector<Minutia> minutiae;
  for (int i = 0; i < n; ++i) {
    int x, y;
    float theta;
    stream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }

  auto area = buildValidArea(minutiae, width, height);
  vector<char> values, validities, cylinderValidities;
  buildTemplate(minutiae, area, width, height,
    cylinderValidities, values, validities);
  handleError(cudaDeviceSynchronize());

  vector<char> dummy;
  debug("Global score: %f\n", matchTemplate(
    cylinderValidities, validities, values,
    cylinderValidities, validities, values));

  return 0;
}
