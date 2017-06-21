#include <chrono>
#include <iostream>

#include "mcc.cuh"
#include "constants.cuh"
#include "minutia.cuh"
#include "io.cuh"
#include "area.cuh"
#include "template.cuh"
#include "matcher.cuh"
#include "errors.h"

using namespace std;

MCC::MCC(const char *input):
  input(input), loaded(false), built(false) {
}

MCC::~MCC() {
}

bool MCC::load() {
  if (loaded) return true;
  return loaded = loadMinutiaeFromFile(input, &width, &height, &dpi, &n, minutiae);
}

bool MCC::build() {
  if (built) return true;
  if (!loaded) return false;

  size_t devMinutiaeSize = minutiae.size() * sizeof(Minutia);
  handleError(
    cudaMalloc(&devMinutiae, devMinutiaeSize));
  handleError(
    cudaMemcpy(devMinutiae, minutiae.data(), devMinutiaeSize, cudaMemcpyHostToDevice));

  size_t devAreaSize = width * height * sizeof(char);
  handleError(
    cudaMalloc(&devArea, devAreaSize));

  size_t devCylinderValiditiesSize = minutiae.size() * sizeof(char);
  handleError(
    cudaMalloc(&devCylinderValidities, devCylinderValiditiesSize));

  size_t devCellValiditiesSize = minutiae.size() * NC * sizeof(char);
  handleError(
    cudaMalloc(&devCellValidities, devCellValiditiesSize));

  size_t devCellValuesSize = minutiae.size() * NC * sizeof(char);
  handleError(
    cudaMalloc(&devCellValues, devCellValuesSize));

  auto begin = std::chrono::high_resolution_clock::now();
  devBuildValidArea(minutiae, width, height, devArea);
  devBuildTemplate(
    devMinutiae, minutiae.size(),
    devArea, width, height,
    devCylinderValidities,
    devCellValidities,
    devCellValues);
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(end-begin).count();
  cout << "Time taken to build template: " << duration << " microseconds\n";

  return built = true;
}

void MCC::dispose() {
  if (!built) return;

  cudaFree(devMinutiae);
  cudaFree(devArea);
  cudaFree(devCylinderValidities);
  cudaFree(devCellValidities);
  cudaFree(devCellValues);

  built = false;
}

bool MCC::match(const char *target,
    float &similarity, int &n, int &m, vector<float> &matrix) {
  MCC mcc(target);
  if (!mcc.load() || !mcc.build()) return false;

  auto begin = std::chrono::high_resolution_clock::now();
  similarity = devMatchTemplate(
    devMinutiae, minutiae.size(),
    devCylinderValidities, devCellValidities, devCellValues,
    mcc.devMinutiae, mcc.minutiae.size(),
    mcc.devCylinderValidities, mcc.devCellValidities, mcc.devCellValues,
    matrix);
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(end-begin).count();
  cout << "Time taken to match templates: " << duration << " microseconds\n";

  n = minutiae.size();
  m = mcc.minutiae.size();

  mcc.dispose();
  return true;
}
