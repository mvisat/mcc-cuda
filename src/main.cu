#include <vector>
#include <iostream>

#include "errors.h"
#include "debug.h"
#include "constants.cuh"
#include "template.cuh"
#include "matcher.cuh"
#include "io.cuh"
#include "mcc.cuh"
#include "consolidation.cuh"

using namespace std;

bool buildTemplateFromFile(
    const char *input,
    const char *output) {
  int width, height, dpi, n;
  vector<Minutia> minutiae;
  if (!loadMinutiaeFromFile(input, width, height, dpi, n, minutiae))
    return false;

  vector<char> cylinderValidities, cellValidities, cellValues;
  buildTemplate(minutiae, width, height,
    cylinderValidities, cellValidities, cellValues);
  handleError(cudaDeviceSynchronize());

  return saveTemplateToFile(
    output, width, height, dpi, n, minutiae,
    cylinderValidities.size(), cylinderValidities, cellValidities, cellValues);
}

bool buildSimilarityFromTemplate(
    const char *template1,
    const char *template2,
    const char *output) {
  int width1, height1, dpi1, n1;
  vector<Minutia> minutiae1;
  int m1;
  vector<char> cylinderValidities1, cellValidities1, cellValues1;
  if (!loadTemplateFromFile(template1,
      width1, height1, dpi1, n1, minutiae1,
      m1, cylinderValidities1, cellValidities1, cellValues1))
    return false;

  int width2, height2, dpi2, n2;
  vector<Minutia> minutiae2;
  int m2;
  vector<char> cylinderValidities2, cellValidities2, cellValues2;
  if (!loadTemplateFromFile(template2,
      width2, height2, dpi2, n2, minutiae2,
      m2, cylinderValidities2, cellValidities2, cellValues2))
    return false;

  vector<float> matrix;
  matchTemplate(
    minutiae1, cylinderValidities1, cellValidities1, cellValues1,
    minutiae2, cylinderValidities2, cellValidities2, cellValues2,
    matrix);
  auto similarity = LSS(matrix, m1, m2);
  printf("Similarity: %f\n", similarity);
  return saveSimilarityToFile(output, m1, m2, matrix);
}

bool buildSimilarityFromMinutiae(
    const char *minutiae1,
    const char *minutiae2,
    const char *output) {
  MCC mcc(minutiae1);
  if (!mcc.load() || !mcc.build()) return false;

  float similarity;
  int n, m;
  vector<float> matrix;
  bool ret = mcc.match(minutiae2, similarity, n, m, matrix);
  mcc.dispose();
  if (!ret) return false;
  printf("Similarity: %f\n", similarity);
  return saveSimilarityToFile(output, n, m, matrix);
}

void printUsage(char const *argv[]) {
  cerr << "usage: " << argv[0] << " [mcc|template|match] [options]\n";
  cerr << endl;
  cerr << "mcc\t\t: <in:minutia1> <in:minutia2> <out:similarity>\n";
  cerr << "template\t: <in:minutia> <out:template>\n";
  cerr << "match\t\t: <in:template1> <in:template2> <out:similarity>\n";
}

int main(int argc, char const *argv[]) {
  if (argc > 1) {
    if (strncmp(argv[1], "mcc", 3) == 0 && argc == 5) {
      return !buildSimilarityFromMinutiae(argv[2], argv[3], argv[4]);
    } else if (strncmp(argv[1], "template", 8) == 0 && argc == 4) {
      return !buildTemplateFromFile(argv[2], argv[3]);
    } else if (strncmp(argv[1], "match", 5) == 0 && argc == 5) {
      return !buildSimilarityFromTemplate(argv[2], argv[3], argv[4]);
    }
  }

  printUsage(argv);
  return 1;
}
