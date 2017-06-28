#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include "minutia.cuh"
#include "constants.cuh"

#define ceilMod(x,y) (x+y-1)/y

__host__ __device__ __inline__
bool floatEqual(float a, float b) {
  return fabsf(a - b) < EPS;
}

__host__ __device__ __inline__
bool floatGreater(float a, float b) {
  return (a - b) > ((fabsf(a) < fabsf(b) ? fabsf(b) : fabsf(a)) * EPS);
}

__host__ __device__ __inline__
int sqrDistance(int x1, int y1, int x2, int y2) {
  int dx = x1 - x2;
  int dy = y1 - y2;
  return dx*dx + dy*dy;
}

__host__ __device__ __inline__
int sqrDistanceFromSegment(int x, int y, int x1, int y1, int x2, int y2) {
  auto dist = sqrDistance(x1, y1, x2, y2);
  if (dist == 0) return sqrDistance(x, y, x1, y1);
  auto p = (x-x1)*(x2-x1) + (y-y1)*(y2-y1);
  float t = fmaxf(0.0f, fminf(1.0f, (float)p/dist));
  return sqrDistance(x, y, x1 + t * (x2-x1), y1 + t * (y2-y1));
}

__host__ __device__ __inline__
float distance(int x1, int y1, int x2, int y2) {
  return sqrtf(sqrDistance(x1, y1, x2, y2));
}

__host__ __device__ __inline__
void pointsToLines(int x1, int y1, int x2, int y2, float *a, float *b, float *c) {
  if (floatEqual(x1, x2)) {
    *a = 1.0f;
    *b = 0.0f;
    *c = -x1;
  } else {
    *a = -(float)(y1-y2)/(x1-x2);
    *b = 1.0f;
    *c = -(float)(*a * x1) - y1;
  }
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
float angle(float theta1, float theta2) {
  float diff = theta1-theta2;
  if (diff < -M_PI)
    return M_2PI + diff;
  if (diff >= M_PI)
    return -M_2PI + diff;
  return diff;
}

__host__ __device__ __inline__
float sigmoid(int value, float tau, float mu) {
  return 1.0f / (1.0f + expf(-tau * (value-mu)));
}

#endif
