#include <vector>
#include <iostream>

#include "errors.h"
#include "debug.h"
#include "constants.cuh"
#include "template.cuh"
#include "matcher.cuh"
#include "io.cuh"

using namespace std;

bool buildTemplateFromFile(
    const char *input,
    const char *output) {
  int width, height, dpi, n;
  vector<Minutia> minutiae;
  if (!loadMinutiaeFromFile(input, &width, &height, &dpi, &n, minutiae))
    return false;

  vector<char> cylinderValidities, cellValidities, cellValues;
  buildTemplate(minutiae, width, height,
    cylinderValidities, cellValidities, cellValues);
  handleError(cudaDeviceSynchronize());

  return saveTemplateToFile(
    output, width, height, dpi, n, minutiae,
    cylinderValidities.size(), cylinderValidities, cellValidities, cellValues);
}

bool buildSimilarityFromFile(
    const char *template1,
    const char *template2,
    const char *output) {
  int width1, height1, dpi1, n1;
  vector<Minutia> minutiae1;
  int m1;
  vector<char> cylinderValidities1, cellValidities1, cellValues1;
  if (!loadTemplateFromFile(template1,
      &width1, &height1, &dpi1, &n1, minutiae1,
      &m1, cylinderValidities1, cellValidities1, cellValues1))
    return false;

  int width2, height2, dpi2, n2;
  vector<Minutia> minutiae2;
  int m2;
  vector<char> cylinderValidities2, cellValidities2, cellValues2;
  if (!loadTemplateFromFile(template2,
      &width2, &height2, &dpi2, &n2, minutiae2,
      &m2, cylinderValidities2, cellValidities2, cellValues2))
    return false;

  vector<float> matrix;
  auto similarity = matchTemplate(
    minutiae1, cylinderValidities1, cellValidities1, cellValues1,
    minutiae2, cylinderValidities2, cellValidities2, cellValues2,
    matrix);
  printf("Similarity: %f\n", similarity);
  return saveSimilarityToFile(output, m1, m2, matrix);
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
      return !buildTemplateFromFile(argv[2], argv[3]);
    } else if (strncmp(argv[1], "match", 5) == 0 && argc == 5) {
      return !buildSimilarityFromFile(argv[2], argv[3], argv[4]);
    }
  }

  printUsage(argv);
  return 1;
}
