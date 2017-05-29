#include <fstream>
#include <vector>
#include <iostream>

#include "errors.h"
#include "template.cuh"
#include "constants.cuh"
#include "area.cuh"

using namespace std;

int main() {
  int rows, cols, dpi, n;
  ifstream stream("data/1_1.txt");
  stream >> rows >> cols >> dpi >> n;
  vector<Minutia> minutiae;
  for (int i = 0; i < n; ++i) {
    int x, y;
    float theta;
    stream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }

  vector<char> x = buildValidArea(minutiae, rows, cols);
  handleError(
    cudaDeviceSynchronize());
  return 0;
}
