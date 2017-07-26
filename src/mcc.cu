#include <chrono>
#include <iostream>
#include <thread>
#include <functional>
#include <algorithm>

#include "mcc.cuh"
#include "constants.cuh"
#include "minutia.cuh"
#include "io.cuh"
#include "area.cuh"
#include "template.cuh"
#include "binarization.cuh"
#include "matcher.cuh"
#include "consolidation.cuh"
#include "errors.h"

using namespace std;

MCC::MCC() {
  initialize();
}

MCC::MCC(const char *input, bool autoLoad):
    input(input) {
  initialize();
  if (autoLoad) {
    load();
    build();
  }
}

MCC::~MCC() {
  dispose();
}

void MCC::initialize() {
  devMinutiae = NULL;
  devArea = NULL;
  devCylinderValidities = NULL;
  devCellValidities = NULL;
  devCellValues = NULL;
  devBinarizedValidities = NULL;
  devBinarizedValues = NULL;
  devMatrix = NULL;
  allocate();
}

void MCC::allocate() {
  size_t devMinutiaeSize = MAX_MINUTIAE * sizeof(Minutia);
  handleError(
    cudaMalloc(&devMinutiae, devMinutiaeSize));

  size_t devAreaSize = MAX_WIDTH * MAX_HEIGHT * sizeof(char);
  handleError(
    cudaMalloc(&devArea, devAreaSize));

  size_t devCylinderValiditiesSize = MAX_MINUTIAE * sizeof(char);
  handleError(
    cudaMalloc(&devCylinderValidities, devCylinderValiditiesSize));

  size_t devCellSize = MAX_MINUTIAE * NC * sizeof(char);
  handleError(
    cudaMalloc(&devCellValidities, devCellSize));
  handleError(
    cudaMalloc(&devCellValues, devCellSize));

  int intPerCylinder = NC/BITS;
  size_t devBinarizedSize = MAX_MINUTIAE * intPerCylinder * sizeof(unsigned int);
  handleError(
    cudaMalloc(&devBinarizedValidities, devBinarizedSize));
  handleError(
    cudaMalloc(&devBinarizedValues, devBinarizedSize));

  size_t devMatrixSize = MAX_MINUTIAE * MAX_MINUTIAE * sizeof(float);
  handleError(
    cudaMalloc(&devMatrix, devMatrixSize));
}

void MCC::dispose() {
  if (devMinutiae != NULL) {
    cudaFree(devMinutiae);
    devMinutiae = NULL;
  }
  if (devArea != NULL) {
    cudaFree(devArea);
    devArea = NULL;
  }
  if (devCylinderValidities != NULL) {
    cudaFree(devCylinderValidities);
    devCylinderValidities = NULL;
  }
  if (devCellValidities != NULL) {
    cudaFree(devCellValidities);
    devCellValidities = NULL;
  }
  if (devCellValues != NULL) {
    cudaFree(devCellValues);
    devCellValues = NULL;
  }
  if (devBinarizedValidities != NULL) {
    cudaFree(devBinarizedValidities);
    devBinarizedValidities = NULL;
  }
  if (devBinarizedValues != NULL) {
    cudaFree(devBinarizedValues);
    devBinarizedValues = NULL;
  }
  if (devMatrix != NULL) {
    cudaFree(devMatrix);
    devMatrix = NULL;
  }
}

bool MCC::load() {
  return loadMinutiaeFromFile(input, width, height, dpi, n, minutiae);
}

bool MCC::load(const char *input) {
  this->input = input;
  return load();
}

bool MCC::build() {
  size_t devMinutiaeSize = minutiae.size() * sizeof(Minutia);
  handleError(
    cudaMemcpy(devMinutiae, minutiae.data(), devMinutiaeSize, cudaMemcpyHostToDevice));

  devBuildValidArea(minutiae, width, height, devArea);
  devBuildTemplate(
    devMinutiae, minutiae.size(),
    devArea, width, height,
    devCylinderValidities,
    devCellValidities,
    devCellValues);
  devBinarizedTemplate(minutiae.size(),
    devCellValidities, devCellValues,
    devBinarizedValidities, devBinarizedValues);
  return true;
}

bool MCC::match(const char *target,
    float &similarity, int &n, int &m, vector<float> &matrix) {
  MCC mcc(target, false);
  if (!mcc.load() || !mcc.build()) return false;

  n = minutiae.size();
  m = mcc.minutiae.size();

  auto begin = std::chrono::high_resolution_clock::now();
  devMatchTemplate(
    devMinutiae, n,
    devCylinderValidities, devBinarizedValidities, devBinarizedValues,
    mcc.devMinutiae, m,
    mcc.devCylinderValidities, mcc.devBinarizedValidities, mcc.devBinarizedValues,
    devMatrix);
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(end-begin).count();
  cout << "Time taken to match templates: " << duration << " microseconds\n";

  size_t devMatrixSize = n * m * sizeof(float);
  matrix.resize(n * m);
  handleError(
    cudaMemcpy(matrix.data(), devMatrix, devMatrixSize, cudaMemcpyDeviceToHost));

  begin = std::chrono::high_resolution_clock::now();
  similarity = LSSR(matrix, n, m, minutiae, mcc.minutiae);
  end = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<chrono::microseconds>(end-begin).count();
  cout << "Time taken to compute global score: " << duration << " microseconds\n";

  return true;
}

void MCC::matchMany(const vector<string> &targets, vector<float> &values) {
  int numThreads = thread::hardware_concurrency();
  thread threads[numThreads];

  auto begin = std::chrono::high_resolution_clock::now();
  load();
  build();
  for (int t = 0; t < numThreads; ++t) {
    threads[t] = thread([&](int tid) {
      int n = minutiae.size();
      MCC mcc;
      for (int i = tid; i < targets.size(); i += numThreads) {
        mcc.load(targets[i].c_str());
        mcc.build();
        int m = mcc.minutiae.size();

        devMatchTemplate(
          devMinutiae, n,
          devCylinderValidities, devBinarizedValidities, devBinarizedValues,
          mcc.devMinutiae, m,
          mcc.devCylinderValidities, mcc.devBinarizedValidities, mcc.devBinarizedValues,
          mcc.devMatrix);

        size_t devMatrixSize = n * m * sizeof(float);
        vector<float> matrix(n * m);
        handleError(
          cudaMemcpy(matrix.data(), mcc.devMatrix, devMatrixSize, cudaMemcpyDeviceToHost));

        values[i] = LSSR(matrix, n, m, minutiae, mcc.minutiae);
      }
    }, t);
  }
  for (int i = 0; i < numThreads; ++i) {
    threads[i].join();
  }
  handleError(
    cudaDeviceSynchronize());
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end-begin).count();

  auto maxv = values.front();
  auto maxi = 0;
  for (int i = 1; i < values.size(); ++i) {
    if (values[i] > maxv) {
      maxv = values[i];
      maxi = i;
    }
  }
  cout << duration << " " << input << " " << targets[maxi] << " " << maxv << endl;
  // cout << "Input " << input << " best match: " << targets[maxi] << " " << maxv << endl;
}
