#include <cstdlib>
#include <algorithm>

#include "minutia.cuh"
#include "constants.cuh"
#include "errors.h"

using namespace std;

__host__ __device__ __inline__
int sqr_distance(int x1, int y1, int x2, int y2) {
  int dx = x1-x2;
  int dy = y1-y2;
  return dx*dx - dy*dy;
}

/* Check line's turn created by 3 points (a -> b -> c)
 *-1: turn right (clockwise)
 * 0: collinear
 * 1: turn left (counter clockwise)
 */
__host__ __device__ __inline__
int line_turn(int ax, int ay, int bx, int by, int cx, int cy) {
  int pos = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
  if (pos > 0) return 1;
  if (pos < 0) return -1;
  return 0;
}

__host__ __device__ __inline__
int minutiae_turn(const Minutia& a, const Minutia& b, const Minutia& c) {
  return line_turn(a.x, a.y, b.x, b.y, c.x, c.y);
}

__host__
vector<Minutia> buildConvexHull(vector<Minutia>& minutiae) {
  int min_y = 0;
  for (int i = 1; i < minutiae.size(); ++i) {
    if (minutiae[i].y < minutiae[min_y].y)
      min_y = i;
  }

  Minutia pivot = minutiae[min_y];
  swap(minutiae[0], minutiae[min_y]);
  sort(begin(minutiae)+1, end(minutiae), [&]
      (const Minutia &lhs, const Minutia &rhs) {
    int turn = minutiae_turn(pivot, lhs, rhs);
    if (turn == 0) {
      auto ldist = sqr_distance(pivot.x, pivot.y, lhs.x, lhs.y);
      auto rdist = sqr_distance(pivot.x, pivot.y, rhs.x, rhs.y);
      return ldist < rdist;
    }
    return turn == 1;
  });

  vector<Minutia> hull;
  for (int i = 0; i < 3; ++i)
    hull.push_back(minutiae[i]);

  for (int i = 3; i < minutiae.size(); ++i) {
    Minutia top = hull.back();
    while (minutiae_turn(hull.back(), top, minutiae[i]) != 1) {
      top = hull.back();
      hull.pop_back();
    }
    hull.push_back(top);
    hull.push_back(minutiae[i]);
  }
  return hull;
}

__global__
void fillConvexHull(Minutia *hull, const int nHull, char *area) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int idx = x * blockDim.x + y;

  for (int i = 0; i < nHull; ++i) {
    int b = (i+1) % nHull;
    if (line_turn(hull[i].x, hull[i].y, hull[b].x, hull[b].y, x, y) < 0) {
      area[idx] = 0;
      return;
    }
  }
  area[idx] = 1;
}

__global__
void extendConvexHull(char *area, int rows, int cols, int radius, char *extended) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int idx = x * cols + y;

  if (extended[idx])
    return;

  if (area[idx]) {
    extended[idx] = 1;
    return;
  }

  int i = x - radius + threadIdx.x;
  if (i < 0 || i >= rows)
    return;

  const int radius2 = radius*radius;
  int stride = i * cols;
  for (int j = y-radius; j < y+radius; ++j) {
    if (j < 0 || j >= cols || !area[stride + j]) continue;
    if (sqr_distance(x, y, i, j) <= radius2) {
      extended[idx] = 1;
      return;
    }
  }
}

__host__
vector<char> buildValidArea(vector<Minutia>& minutiae, int rows, int cols) {
  vector<Minutia> hull = buildConvexHull(minutiae);

  size_t devHullSize = hull.size() * sizeof(Minutia);
  size_t devAreaSize = rows * cols * sizeof(char);
  Minutia *devHull = nullptr;
  char *devArea = nullptr;
  handleError(
    cudaMalloc(&devHull, devHullSize));
  handleError(
    cudaMemcpy(devHull, hull.data(), devHullSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devArea, rows * cols * sizeof(char)));
  fillConvexHull<<<rows, cols>>>(devHull, hull.size(), devArea);
  cudaFree(devHull);

  char *devExtended = nullptr;
  handleError(
    cudaMalloc(&devExtended, devAreaSize));
  handleError(
    cudaMemset(devExtended, 0, devAreaSize));
  dim3 gridDim(rows, cols);
  dim3 blockDim(2*OMEGA);
  extendConvexHull<<<gridDim, blockDim>>>(devArea, rows, cols, OMEGA, devExtended);
  cudaFree(devArea);

  vector<char> ret(rows * cols);
  handleError(
    cudaMemcpy(ret.data(), devExtended, devAreaSize, cudaMemcpyDeviceToHost));
  cudaFree(devExtended);

  return ret;
}
