#include <algorithm>
#include <tuple>
#include <fstream>

#include "minutia.cuh"
#include "constants.cuh"
#include "util.cuh"
#include "errors.h"
#include "debug.h"

using namespace std;

__host__
vector<Minutia> buildConvexHull(const vector<Minutia>& _minutiae) {
  vector<Minutia> minutiae(_minutiae);
  int min_y = 0;
  for (int i = 1; i < minutiae.size(); ++i) {
    if (minutiae[i] < minutiae[min_y])
      min_y = i;
  }

  Minutia pivot = minutiae[min_y];
  swap(minutiae.front(), minutiae[min_y]);
  sort(minutiae.begin()+1, minutiae.end(), [&]
      (const Minutia &lhs, const Minutia &rhs) {
    int turn = minutiaTurn(pivot, lhs, rhs);
    if (turn == 0) {
      auto ldist = sqrDistance(pivot.x, pivot.y, lhs.x, lhs.y);
      auto rdist = sqrDistance(pivot.x, pivot.y, rhs.x, rhs.y);
      return ldist < rdist;
    }
    return turn == 1;
  });

  vector<Minutia> hull;
  for (int i = 0; i < 3; ++i)
    hull.push_back(minutiae[i]);

  for (int i = 3; i < minutiae.size(); ++i) {
    Minutia top = hull.back();
    while (minutiaTurn(hull.back(), top, minutiae[i]) != 1) {
      top = hull.back();
      hull.pop_back();
    }
    hull.push_back(top);
    hull.push_back(minutiae[i]);
  }
  return hull;
}


__global__
void fillConvexHull(Minutia *hull, const int nHull,
    const int width, const int height, char *area) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  extern __shared__ int shared[];
  Minutia *sharedHull = (Minutia*)shared;
  int idx = threadIdx.y * blockDim.x + threadIdx.x;
  if (idx < nHull) sharedHull[idx] = hull[idx];
  __syncthreads();

  char ok = 1;
  for (int i = 0; i < nHull; ++i) {
    Minutia a(sharedHull[i]);
    Minutia b(sharedHull[(i+1) % nHull]);
    if (lineTurn(x, y, a.x, a.y, b.x, b.y) < 0) {
      ok = 0;
      if (sqrDistanceFromSegment(x, y, a.x, a.y, b.x, b.y) <= OMEGA*OMEGA) {
        area[y*width + x] = 1;
        return;
      }
    }
  }
  area[y*width + x] = ok;
}

__host__
void devBuildValidArea(
    const vector<Minutia> &minutiae,
    const int width, const int height,
    char *devArea) {
  auto hull = buildConvexHull(minutiae);

#ifdef DEBUG
  ofstream hullStream("plot/hull.txt");
  hullStream << width << endl << height << endl << hull.size() << endl;
  for (int i = 0; i < hull.size(); ++i)
    hullStream << hull[i].x << ' ' << hull[i].y << endl;
  hullStream << endl;
  hullStream.close();
#endif

  Minutia *devHull;
  size_t devHullSize = hull.size() * sizeof(Minutia);
  handleError(
    cudaMalloc(&devHull, devHullSize));
  handleError(
    cudaMemcpy(devHull, hull.data(), devHullSize, cudaMemcpyHostToDevice));

  int threadPerDim = 32;
  dim3 blockCount(ceilMod(width, threadPerDim), ceilMod(height, threadPerDim));
  dim3 threadCount(threadPerDim, threadPerDim);
  fillConvexHull<<<blockCount, threadCount, devHullSize>>>(
    devHull, hull.size(), width, height, devArea);
  handleError(
    cudaPeekAtLastError());

  cudaFree(devHull);
}

__host__
vector<char> buildValidArea(
    const vector<Minutia>& minutiae,
    const int width, const int height) {

  char *devArea;
  size_t devAreaSize = width * height * sizeof(char);
  handleError(
    cudaMalloc(&devArea, devAreaSize));

  devBuildValidArea(minutiae, width, height, devArea);

  vector<char> ret(width * height);
  handleError(
    cudaMemcpy(ret.data(), devArea, devAreaSize, cudaMemcpyDeviceToHost));

  cudaFree(devArea);

#ifdef DEBUG
  ofstream areaStream("plot/area.txt");
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      areaStream << (ret[i*width + j] ? '1' : '0');
    }
    areaStream << endl;
  }
  areaStream.close();
#endif

  return ret;
}
