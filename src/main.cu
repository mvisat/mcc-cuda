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

void buildTemplateFromFile(
    const char *fileName,
    vector<char>& cylinderValidities,
    vector<char>& cellValues,
    vector<char>& cellValidities) {
  int width, height, dpi, n;
  ifstream stream(fileName);
  stream >> width >> height >> dpi >> n;
  vector<Minutia> minutiae;
  for (int i = 0; i < n; ++i) {
    int x, y;
    float theta;
    stream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }

  auto area = buildValidArea(minutiae, width, height);
  buildTemplate(minutiae, area, width, height,
    cylinderValidities, cellValues, cellValidities);
  handleError(cudaDeviceSynchronize());
}

int main() {
  vector<char> cellValues1, cellValidities1, cylinderValidities1;
  buildTemplateFromFile(
    "data/1_1.txt", cylinderValidities1, cellValidities1, cellValues1);

  debug("Cylinder validities:\n");
  for (int i = 0; i < cylinderValidities1.size(); ++i)
    debug("%d ", cylinderValidities1[i]);
  debug("\n\n");

  debug("Cell validities:\n");
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      debug("%d ", cellValidities1[2*NC + i*NS*ND + j*ND]);
    }
  }
  debug("\n\n");

  debug("Cell values:\n");
  for (int i = 0; i < NC; ++i)
    debug("%d ", cellValues1[2*NC + i]);
  debug("\n\n");

  // vector<char> cellValues2, cellValidities2, cylinderValidities2;
  // buildTemplateFromFile(
  //   "data/1_2.txt", cylinderValidities2, cellValidities2, cellValues2);
  //
  // auto globalScore = matchTemplate(
  //   cylinderValidities1, cellValidities2, cellValues1,
  //   cylinderValidities2, cellValidities1, cellValues2);
  // debug("Global score: %f\n", globalScore);

  return 0;
}
