#include <cstdlib>
#include <cstdio>
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

  ofstream of("tes.txt");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int idx = i * cols + j;
      of << (x[idx] ? '1' : '0');
    }
    of << endl;
  }
  return 0;
}
