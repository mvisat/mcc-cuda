#include <algorithm>
#include <tuple>
#include <fstream>

#include "minutia.cuh"
#include "constants.cuh"
#include "errors.h"
#include "debug.h"

#define ceilMod(x,y) (x+y-1)/y

struct Point {
  int x, y;
  Point(int x, int y): x(x), y(y) {};
};

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


__host__
vector<Point> extendConvexHull(const vector<Minutia>& hull,
    const int width, const int height, const int radius) {
  vector<Point> extended;
  for (int i = 1; i <= hull.size(); ++i) {
    Minutia p1(hull[i-1]);
    Minutia p2(hull[i%hull.size()]);
    int dx = p2.x-p1.x, dy = p2.y-p1.y;
    float rdist = rsqrtf(dx*dx + dy*dy);
    int dxr = (dy * radius) * rdist;
    int dyr = (-dx * radius) * rdist;
    extended.emplace_back(p1.x, p1.y);
    extended.emplace_back(p1.x + dxr, p1.y + dyr);
    extended.emplace_back(p2.x + dxr, p2.y + dyr);
    extended.emplace_back(p2.x, p2.y);
  }
  return extended;
}

__global__
void fillConvexHull(Minutia *hull, Point *boxes, const int nHull,
    const int width, const int height, char *area) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // fill original convex hull
  bool ok = true;
  for (int i = 0; i < nHull; ++i) {
    int b = (i+1) % nHull;
    if (lineTurn(hull[i].x, hull[i].y, hull[b].x, hull[b].y, x, y) < 0) {
      ok = false;
      break;
    }
  }
  if (ok) {
    area[y*width + x] = 1;
    return;
  }

  // fill extended area
  for (int i = 0; i < nHull; ++i) {
    bool ok = true;
    for (int j = 0; j < 4; ++j) {
      int k = i*4 + j;
      int l = i*4 + (j+1)%4;
      if (lineTurn(boxes[k].x, boxes[k].y, boxes[l].x, boxes[l].y, x, y) < 0) {
        ok = false;
        break;
      }
    }
    if (ok) {
      area[y*width + x] = 1;
      return;
    }
  }
}

__global__
void fillHullCircle(Minutia *hull, const int nHull, const int radius,
    const int width, const int height, char *area) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x - radius;
  int dy = blockIdx.y * blockDim.y + threadIdx.y - radius;
  if (dx*dx + dy*dy > radius*radius) return;

  for (int i = 0; i < nHull; ++i) {
    int x = hull[i].x + dx;
    int y = hull[i].y + dy;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    area[y*width + x] = 1;
  }
}

__host__
vector<char> buildValidArea(const vector<Minutia>& minutiae,
    const int width, const int height) {
  auto hull = buildConvexHull(minutiae);
  auto boxes = extendConvexHull(hull, width, height, OMEGA);

  size_t devHullSize = hull.size() * sizeof(Minutia);
  size_t devBoxesSize = boxes.size() * sizeof(Point);
  size_t devAreaSize = width * height * sizeof(char);
  Minutia *devHull;
  Point *devBoxes;
  char *devArea;
  handleError(
    cudaMalloc(&devHull, devHullSize));
  handleError(
    cudaMemcpy(devHull, hull.data(), devHullSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devBoxes, devBoxesSize));
  handleError(
    cudaMemcpy(devBoxes, boxes.data(), devBoxesSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devArea, devAreaSize));
  handleError(
    cudaMemset(devArea, 0, devAreaSize));

  int threadPerDim = 32;
  dim3 blockCount(ceilMod(width, threadPerDim), ceilMod(height, threadPerDim));
  dim3 circleBlockCount(
    ceilMod(2*OMEGA, threadPerDim),
    ceilMod(2*OMEGA, threadPerDim));
  dim3 threadCount(threadPerDim, threadPerDim);

  fillConvexHull<<<blockCount, threadCount>>>(
    devHull, devBoxes, hull.size(), width, height, devArea);
  fillHullCircle<<<circleBlockCount, threadCount>>>(
    devHull, hull.size(), OMEGA, width, height, devArea);

  vector<char> ret(width * height);
  handleError(
    cudaMemcpy(ret.data(), devArea, devAreaSize, cudaMemcpyDeviceToHost));
  cudaFree(devHull);
  cudaFree(devBoxes);
  cudaFree(devArea);

#ifdef DEBUG
  ofstream hullStream("plot/hull.txt");
  hullStream << width << endl << height << endl << hull.size() << endl;
  for (int i = 0; i < hull.size(); ++i)
    hullStream << hull[i].x << ' ' << hull[i].y << endl;
  hullStream << endl;
  hullStream.close();

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
