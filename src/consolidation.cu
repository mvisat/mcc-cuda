#include <algorithm>
#include <functional>

#include "consolidation.cuh"
#include "constants.cuh"
#include "util.cuh"
#include "sort.cuh"
#include "debug.h"

using namespace std;

__host__ __device__ __inline__
int getNP(const int rows, const int cols) {
  return MIN_NP +
    roundf(sigmoid(min(rows, cols), TAU_P, MU_P) * (MAX_NP - MIN_NP));
}

__host__
float LSS(const vector<float>& _matrix, const int rows, const int cols) {
  auto matrix(_matrix);
  int np = getNP(rows, cols);
  nth_element(matrix.begin(), matrix.begin()+np, matrix.end(), greater<float>());
  auto sum = accumulate(matrix.begin(), matrix.begin()+np, 0.0f);
  return sum / np;
}

__host__
float devLSS(float *devMatrix, const int rows, const int cols) {
  int np = getNP(rows, cols);
  devBitonicSort(devMatrix, rows*cols);
  vector<float> matrix(np);
  cudaMemcpy(matrix.data(), devMatrix, np * sizeof(float), cudaMemcpyDeviceToHost);
  auto sum = accumulate(matrix.begin(), matrix.end(), 0.0f);
  return sum / np;
}

__host__
float LSSR(const vector<float> &_gamma,
    const int rows, const int cols,
    const vector<Minutia> &minutiaeA, const vector<Minutia> &minutiaeB) {
  int np = getNP(rows, cols);
  int nr = min(rows, cols);

  vector<tuple<float, int, int>> gamma;
  gamma.reserve(rows*cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      gamma.emplace_back(_gamma[i*cols + j], i, j);
  nth_element(gamma.begin(), gamma.begin()+nr, gamma.end(),
    [](const tuple<float, int, int> &a, const tuple<float, int, int> &b) -> bool {
      return get<0>(a) > get<0>(b);
  });

#ifdef DEBUG
  debug("[LSSR] Pre-selected matrix elements:\n");
  for (int i = 0; i < nr; ++i)
    debug("%d %d\n", get<1>(gamma[i]), get<2>(gamma[i]));
  debug("\n");
#endif

  vector<float> lambda(nr);
  vector<float> lambda1(nr);
  vector<float> rho(nr*nr);
  for (int i = 0; i < nr; ++i) {
    float g; int rt, ct;
    tie(g, rt, ct) = gamma[i];
    lambda[i] = g;
    for (int j = 0; j < nr; ++j) {
      int idx = i*nr + j;
      if (i == j) {
        rho[idx] = 0.0f;
      } else {
        int rk, ck;
        tie(g, rk, ck) = gamma[j];
      	float d1 = fabsf(
          distance(minutiaeA[rt].x, minutiaeA[rt].y, minutiaeA[rk].x, minutiaeA[rk].y)-
          distance(minutiaeB[ct].x, minutiaeB[ct].y, minutiaeB[ck].x, minutiaeB[ck].y));
      	float d2 = fabsf(angle(
          angle(minutiaeA[rt].theta, minutiaeA[rk].theta),
          angle(minutiaeB[ct].theta, minutiaeB[ck].theta)));
      	float d3 = fabsf(angle(
          radian(minutiaeA[rt], minutiaeA[rk]),
          radian(minutiaeB[ct], minutiaeB[ck])));
        rho[idx] =
          sigmoid(d1, TAU_P1, MU_P1) *
          sigmoid(d2, TAU_P2, MU_P2) *
          sigmoid(d3, TAU_P3, MU_P3);
      }
    }
  }

  const float WL = (1.0f - WR) / (nr - 1);
  for (int rel = 0; rel < N_REL; ++rel) {
    lambda1.swap(lambda);
    for (int i = 0; i < nr; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < nr; ++j) {
        sum += rho[i*nr + j] * lambda1[j];
      }
      lambda[i] = WR * lambda1[i] + WL * sum;
    }
  }

#ifdef DEBUG
  vector<tuple<int,int,float>> relaxed(nr);
  for (int i = 0; i < nr; ++i)
    relaxed[i] = make_tuple(get<1>(gamma[i]), get<2>(gamma[i]), lambda[i]);
  sort(relaxed.begin(), relaxed.end(),
    [](const tuple<int,int,float> &a, const tuple<int,int,float> &b) -> bool {
      if (get<0>(a) == get<0>(b))
        return get<1>(a) < get<1>(b);
      return get<0>(a) < get<0>(b);
    });
  debug("[LSSR] Relaxed matrix:\n");
  for (auto &t: relaxed)
    debug("%d %d %f\n", get<0>(t), get<1>(t), get<2>(t));
  debug("\n");
#endif

  vector<tuple<float, int>> efficiencies;
  efficiencies.reserve(nr);
  for (int i = 0; i < nr; ++i)
    efficiencies.emplace_back(lambda[i] / get<0>(gamma[i]), i);
  struct { bool operator()(
    const tuple<float, int> &a, const tuple<float, int> &b) const {
      return get<0>(a) > get<0>(b);
    }
  } cmp;
#ifndef DEBUG
  nth_element(efficiencies.begin(), efficiencies.begin()+np, efficiencies.end(), cmp);
#else
  sort(efficiencies.begin(), efficiencies.end(), cmp);
#endif

#ifdef DEBUG
  debug("[LSSR] Selected matrix elements and efficiencies:\n");
  for (int i = 0; i < nr; ++i) {
    if (i == np) debug("---\n");
    auto idx = get<1>(efficiencies[i]);
    debug("%d %d %f\n", get<1>(gamma[idx]), get<2>(gamma[idx]), get<0>(efficiencies[i]));
  }
  debug("\n");
#endif

  float sum = 0.0f;
  for (int i = 0; i < np; ++i)
    sum += lambda[get<1>(efficiencies[i])];
  return sum / np;
}
