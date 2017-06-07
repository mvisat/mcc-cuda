#include <vector>

#include "errors.h"
#include "minutia.cuh"
#include "constants.cuh"

using namespace std;

__host__ __device__ __inline__
float gaussian(float value) {
  const float GS_DIV = sqrtf(M_2PI) * SIGMA_S;
  return expf(-(value*value)/SIGMA_2S_SQR) / GS_DIV;
}

// http://www.wolframalpha.com/input/?i=integrate+(e%5E(-(t%5E2)%2F(2(x%5E2)))+dt)
__host__ __device__ __inline__
float gaussianIntegral(float value) {
  const float a = sqrtf(M_PI_2) * SIGMA_D;
  const float b = M_SQRT2 * SIGMA_D;
  auto integrate = [&](float val) {
    return a * erff(val/b);
  };
  return rsqrtf(M_2PI) *
    (integrate(value+DELTA_D_2)-integrate(value-DELTA_D_2))
    / SIGMA_D;
}

__host__ __device__ __inline__
int sqrDistance(int x1, int y1, int x2, int y2) {
  int dx = x1 - x2;
  int dy = y1 - y2;
  return dx*dx + dy*dy;
}

__host__ __device__ __inline__
float distance(int x1, int y1, int x2, int y2) {
  return sqrtf(sqrDistance(x1, y1, x2, y2));
}

__host__ __device__ __inline__
float spatialContribution(
    int mt_x, int mt_y, int pi, int pj) {
  return gaussian(distance(mt_x, mt_y, pi, pj));
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
float directionalContribution(
    float m_theta, float mt_theta, float dphik) {
  return gaussianIntegral(
    angle(dphik, angle(m_theta, mt_theta)));
}

__global__
void buildCylinder(
    Minutia *minutiae, char *validArea,
    int rows, int cols, char2 *cells) {
  extern __shared__ int shared[];

  const int N = gridDim.x;
  Minutia *sharedMinutiae = (Minutia*)shared;
  char *contributed = (char*)&sharedMinutiae[N];

  if (blockIdx.x < N) {
    sharedMinutiae[blockIdx.x] = minutiae[blockIdx.x];
    contributed[blockIdx.x] = 0;
  }
  __syncthreads();

  int idxMinutia = blockIdx.x;
  Minutia m = sharedMinutiae[idxMinutia];

  int halfNS = -1 + (NS + 1) >> 1;
  int halfNSi = m.x - halfNS;
  int halfNSj = m.y - halfNS;

  float sint, cost;
  sincosf(m.theta, &sint, &cost);
  int pi = m.x + DELTA_S * (cost * halfNSi + sint * halfNSj);
  int pj = m.y + DELTA_S * (-sint * halfNSi + cost * halfNSj);

  const int SIGMA_9S_SQR = 9 * SIGMA_S_SQR;

  char validity = pi >= 0 && pi < rows && pj >= 0 && pj < cols &&
    validArea[pi * cols + pj] && sqrDistance(m.x, m.y, pi, pj) <= R_SQR;

  int idx = idxMinutia * NC + threadIdx.x * NS * NS + threadIdx.y * NS;
  for (int k = 0; k < ND; ++k, ++idx) {
    char value = 0;

    if (validity) {
      float dphik = -M_PI + (k + 0.5f) * DELTA_D;
      float sum = 0.0f;

      for (int l = 0; l < N; ++l) {
        if (l == idxMinutia)
          continue;

        Minutia mt = sharedMinutiae[l];
        if (sqrDistance(m.x, m.y, mt.x, mt.y) > SIGMA_9S_SQR)
          continue;

        contributed[l] = 1;
        float sContrib = spatialContribution(mt.x, mt.y, pi, pj);
        float dContrib = directionalContribution(m.theta, mt.theta, dphik);
        sum += sContrib * dContrib;
      }

      if (sum >= MU_PSI)
        value = 1;
    }
    cells[idx + k] = make_char2(validity, value);
  }
  __syncthreads();

  // TODO: check cylinder validity
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int sum = 0;
    for (int i = 0; i < N; ++i)
      sum += contributed[i];
  }
}

__host__
vector<char2> buildTemplate(
    const vector<Minutia>& minutiae,
    const vector<char>& validArea,
    int rows, int cols) {
  Minutia *devMinutiae;
  char *devArea;
  char2 *devCells;
  size_t devMinutiaeSize = minutiae.size() * sizeof(Minutia);
  size_t devAreaSize = rows * cols * sizeof(char);
  size_t devCellsSize = minutiae.size() * NC * sizeof(char2);
  handleError(
    cudaMalloc(&devMinutiae, devMinutiaeSize));
  handleError(
    cudaMemcpy(devMinutiae, minutiae.data(), devMinutiaeSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devArea, devAreaSize));
  handleError(
    cudaMemcpy(devArea, validArea.data(), devAreaSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCells, devCellsSize));

  dim3 blockDim(NS, NS);
  int sharedSize = devMinutiaeSize + minutiae.size() * sizeof(char);
  buildCylinder<<<minutiae.size(), blockDim, sharedSize>>>(
    devMinutiae, devArea, rows, cols, devCells);

  vector<char2> ret(minutiae.size() * NC);
  handleError(
    cudaMemcpy(ret.data(), devCells, devCellsSize, cudaMemcpyDeviceToHost));

  cudaFree(devMinutiae);
  cudaFree(devArea);
  cudaFree(devCells);

  return ret;
}
