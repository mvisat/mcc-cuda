#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>

#include "errors.h"
#include "constants.cuh"
#include "area.cuh"
#include "template.cuh"

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
  vector<char> values, validities;
  buildTemplate(minutiae, area, width, height, values, validities);
  handleError(cudaDeviceSynchronize());
  return 0;
}
