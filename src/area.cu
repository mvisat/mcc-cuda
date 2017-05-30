#include <algorithm>
#include <tuple>

#include "minutia.cuh"
#include "constants.cuh"
#include "errors.h"

using namespace std;

__host__ __device__ __inline__
int sqrDistance(int x1, int y1, int x2, int y2) {
  int dx = x1-x2;
  int dy = y1-y2;
  return dx*dx + dy*dy;
}

/* Check line's turn created by 3 points (a -> b -> c)
 *-1: turn right (clockwise)
 * 0: collinear
 * 1: turn left (counter clockwise)
 */
__host__ __device__ __inline__
int lineTurn(int ax, int ay, int bx, int by, int cx, int cy) {
  int pos = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
  if (pos > 0) return 1;
  if (pos < 0) return -1;
  return 0;
}

__host__ __device__ __inline__
int minutiaTurn(const Minutia& a, const Minutia& b, const Minutia& c) {
  return lineTurn(a.x, a.y, b.x, b.y, c.x, c.y);
}

__host__ __device__ __inline__
void pointsToLines(int x1, int y1, int x2, int y2, float *a, float *b, float *c) {
  if (fabsf(x1-x2) < EPS) {
    *a = 1.0f;
    *b = 0.0f;
    *c = -x1;
  } else {
    *a = -(float)(y1-y2)/(x1-x2);
    *b = 1.0f;
    *c = -(float)(*a * x1) - y1;
  }
}

__host__ __device__
bool edgesIntersection(
    int xa1, int ya1, int xa2, int ya2,
    int xb1, int yb1, int xb2, int yb2,
    int *x, int *y) {

  float l1a, l1b, l1c;
  float l2a, l2b, l2c;
  pointsToLines(xa1, ya1, xa2, ya2, &l1a, &l1b, &l1c);
  pointsToLines(xb1, yb1, xb2, yb2, &l2a, &l2b, &l2c);

  // check if parallel
  if (fabsf(l1a-l2a) < EPS && fabsf(l1b-l2b) < EPS)
    return false;

  *x = (l2b * l1c - l1b * l2c) / (l2a * l1b - l1a * l2b);
  // test if vertical line to avoid div by zero
  if (fabsf(l1b) > EPS)
    *y = -(l1a * *x + l1c);
  else
    *y = -(l2a * *x + l2c);
  return true;
}


__host__
vector<Minutia> buildConvexHull(vector<Minutia>& minutiae) {
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

__host__
vector<Minutia> extendConvexHull(vector<Minutia>& hull,
    int rows, int cols, int radius) {
  vector<tuple<int,int,int,int>> edges;
  for (int i = 1; i <= hull.size(); ++i) {
    Minutia *p1 = &hull[i-1];
    Minutia *p2 = &hull[i%hull.size()];
    int dx = p2->x-p1->x, dy = p2->y-p1->y;
    float rdist = rsqrtf(dx*dx + dy*dy);
    int dxr = (dy * radius) * rdist;
    int dyr = (-dx * radius) * rdist;
    edges.emplace_back(
      p1->x + dxr, p1->y + dyr,
      p2->x + dxr, p2->y + dyr);
  }

  vector<Minutia> extended;
  for (int i = 1; i <= hull.size(); ++i) {
    int xa1, ya1, xa2, ya2;
    int xb1, yb1, xb2, yb2;
    tie(xa1, ya1, xa2, ya2) = edges[i-1];
    tie(xb1, yb1, xb2, yb2) = edges[i%hull.size()];
    int xi, yi;
    if (edgesIntersection(
        xa1, ya1, xa2, ya2,
        xb1, yb1, xb2, yb2,
        &xi, &yi)) {
      xi = max(min(xi, rows-1), 0);
      yi = max(min(yi, cols-1), 0);
      extended.emplace_back(xi, yi, hull[i-1].theta);
    }
  }
  return extended;
}

__global__
void fillConvexHull(Minutia *hull, const int nHull, char *area) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int idx = x * blockDim.x + y;

  for (int i = 0; i < nHull; ++i) {
    int b = (i+1) % nHull;
    if (lineTurn(hull[i].x, hull[i].y, hull[b].x, hull[b].y, x, y) < 0) {
      area[idx] = 0;
      return;
    }
  }
  area[idx] = 1;
}

__host__
vector<char> buildValidArea(vector<Minutia>& minutiae, int rows, int cols) {
  vector<Minutia> hull = buildConvexHull(minutiae);
  hull = extendConvexHull(hull, rows, cols, OMEGA);

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

  vector<char> ret(rows * cols);
  handleError(
    cudaMemcpy(ret.data(), devArea, devAreaSize, cudaMemcpyDeviceToHost));
  cudaFree(devArea);

  return ret;
}
