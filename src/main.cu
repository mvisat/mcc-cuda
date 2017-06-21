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

void loadMinutiaeFromFile(
    const char *input,
    int *width, int *height, int *dpi, int *n,
    vector<Minutia> &minutiae) {
  ifstream istream(input);
  istream >> *width >> *height >> *dpi >> *n;
  for (int i = 0; i < *n; ++i) {
    int x, y;
    float theta;
    istream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }
  istream.close();
}

void loadTemplateFromFile(
    const char *input,
    int *width, int *height, int *dpi, int *n,
    vector<Minutia> &minutiae,
    int *m,
    vector<char> &cylinderValidities,
    vector<char> &cellValidities,
    vector<char> &cellValues) {
  ifstream istream(input);
  istream >> *width >> *height >> *dpi >> *n;
  for (int i = 0; i < *n; ++i) {
    int x, y;
    float theta;
    istream >> x >> y >> theta;
    minutiae.emplace_back(x, y, theta);
  }
  istream >> *m;
  cylinderValidities.resize(*m);
  cellValidities.reserve(*m * NC);
  cellValues.reserve(*m * NC);
  for (int l = 0; l < *m; ++l) {
    string s;
    istream >> s;
    cylinderValidities[l] = s.compare("True") == 0 ? 1 : 0;
    if (!cylinderValidities[l]) {
      for (int i = 0; i < NC; ++i) {
        cellValidities.push_back(0);
        cellValues.push_back(0);
      }
      continue;
    }
    for (int i = 0; i < NS; ++i) {
      for (int j = 0; j < NS; ++j) {
        int validity;
        istream >> validity;
        for (int k = 0; k < ND; ++k) {
          cellValidities.push_back(validity);
        }
      }
    }
    for (int i = 0; i < NS; ++i) {
      for (int j = 0; j < NS; ++j) {
        for (int k = 0; k < ND; ++k) {
          int value;
          istream >> value;
          cellValues.push_back(value);
        }
      }
    }
  }
  istream.close();
}

void saveTemplateToFile(
    const char *output,
    int width, int height, int dpi, int n,
    const vector<Minutia> &minutiae,
    int m,
    const vector<char> &cylinderValidities,
    const vector<char> &cellValidities,
    const vector<char> &cellValues) {
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

void saveSimilarityToFile(
    const char *output,
    const int n, const int m,
    const vector<float> &matrix) {
  ofstream ostream(output);
  ostream << n << endl;
  ostream << m << endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      ostream << matrix[i*m+j] << (j == m-1 ? '\n' : ' ');
    }
  }
  ostream.close();
}

void buildTemplateFromFile(
    const char *input,
    const char *output) {
  int width, height, dpi, n;
  vector<Minutia> minutiae;
  loadMinutiaeFromFile(input, &width, &height, &dpi, &n, minutiae);

  vector<char> cylinderValidities, cellValidities, cellValues;
  auto area = buildValidArea(minutiae, width, height);
  buildTemplate(minutiae, area, width, height,
    cylinderValidities, cellValidities, cellValues);
  handleError(cudaDeviceSynchronize());

  saveTemplateToFile(
    output, width, height, dpi, n, minutiae,
    cylinderValidities.size(), cylinderValidities, cellValidities, cellValues);
}

void buildSimilarityFromFile(
    const char *template1,
    const char *template2,
    const char *output) {
  int width1, height1, dpi1, n1;
  vector<Minutia> minutiae1;
  int m1;
  vector<char> cylinderValidities1, cellValidities1, cellValues1;
  loadTemplateFromFile(template1,
    &width1, &height1, &dpi1, &n1, minutiae1,
    &m1, cylinderValidities1, cellValidities1, cellValues1);

  int width2, height2, dpi2, n2;
  vector<Minutia> minutiae2;
  int m2;
  vector<char> cylinderValidities2, cellValidities2, cellValues2;
  loadTemplateFromFile(template2,
    &width2, &height2, &dpi2, &n2, minutiae2,
    &m2, cylinderValidities2, cellValidities2, cellValues2);

  vector<float> matrix;
  auto similarity = matchTemplate(
    cylinderValidities1, cellValidities1, cellValues1,
    cylinderValidities2, cellValidities2, cellValues2,
    matrix);
  printf("Similarity: %f\n", similarity);
  saveSimilarityToFile(output, m1, m2, matrix);
}

void printUsage(char const *argv[]) {
  cerr << "usage: " << argv[0] << " [template|match] [options]\n";
  cerr << endl;
  cerr << "template\t: <input> <output>\n";
  cerr << "match\t\t: <template1> <template2> <output>\n";
}

int main(int argc, char const *argv[]) {
  if (argc > 1) {
    if (strncmp(argv[1], "template", 8) == 0 && argc == 4) {
      buildTemplateFromFile(argv[2], argv[3]);
      return 0;
    } else if (strncmp(argv[1], "match", 5) == 0 && argc == 5) {
      buildSimilarityFromFile(argv[2], argv[3], argv[4]);
      return 0;
    }
  }

  printUsage(argv);
  return 1;
}
