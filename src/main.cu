#include <cstdlib>
#include <cstdio>
#include <cstring>
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
    const char *input,
    const char *output) {
  int width, height, dpi, n;
  ifstream istream(input);
  istream >> width >> height >> dpi >> n;
  vector<Minutia> minutiae;
  for (int i = 0; i < n; ++i) {
    int x, y;
    float theta;
    istream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }
  istream.close();

  vector<char> cylinderValidities, cellValidities, cellValues;

  auto area = buildValidArea(minutiae, width, height);
  buildTemplate(minutiae, area, width, height,
    cylinderValidities, cellValidities, cellValues);
  handleError(cudaDeviceSynchronize());

  ofstream ostream(output);
  ostream << width << endl;
  ostream << height << endl;
  ostream << dpi << endl;
  ostream << n << endl;
  for (auto &m: minutiae)
    ostream << m.x << ' ' << m.y << ' ' << m.theta << endl;
  ostream << n << endl;
  for (int i = 0; i < n; ++i) {
    if (cylinderValidities[i]) {
      ostream << "True ";
      for (int j = 0; j < NS; ++j) {
        for (int k = 0; k < NS; ++k) {
          ostream << (cellValidities[i*NC + j*NS*ND + k*ND] ? "1 " : "0 ");
        }
      }
      for (int j = 0; j < NC; ++j) {
        ostream << (cellValues[i*NC + j] ? '1' : '0');
        if (j != NC-1) ostream << ' ';
      }
    } else {
      ostream << "False";
    }
    ostream << endl;
  }
  ostream.close();
}

void printUsage(char const *argv[]) {
  cerr << "usage: " << argv[0] << " [template|match] [options]\n";
  cerr << endl;
  cerr << "template\t: <input> <output>\n";
  cerr << "match\t\t: <template1> <template2>\n";
}

int main(int argc, char const *argv[]) {
  if (argc > 1) {
    if (strncmp(argv[1], "template", 8) == 0 && argc == 4) {
      buildTemplateFromFile(argv[2], argv[3]);
      return 0;
    } else if (strncmp(argv[1], "match", 5) == 0 && argc == 4) {
      // TODO
      return 0;
    }
  }

  printUsage(argv);
  return 1;
}
