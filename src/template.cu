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
    int width, int height,
    char *cylinderValidities,
    char *cellValues,
    char *cellValidities) {
  extern __shared__ int shared[];

  const int N = gridDim.x;
  char *contributed = (char*)shared;
  Minutia *sharedMinutiae = (Minutia*)&contributed[N];

  if (blockIdx.x < N) {
    sharedMinutiae[blockIdx.x] = minutiae[blockIdx.x];
    contributed[blockIdx.x] = 0;
  }
  __syncthreads();

  int idxMinutia = blockIdx.x;
  Minutia m = sharedMinutiae[idxMinutia];

  float halfNS = (NS + 1) / 2.0f;
  float halfNSi = (threadIdx.x+1) - halfNS;
  float halfNSj = (threadIdx.y+1) - halfNS;
  float sint, cost;
  sincosf(m.theta, &sint, &cost);
  int pi = m.x + DELTA_S * (cost * halfNSi + sint * halfNSj);
  int pj = m.y + DELTA_S * (-sint * halfNSi + cost * halfNSj);

  char validity = pi >= 0 && pi < width && pj >= 0 && pj < height
    && validArea[pj * width + pi]
    && sqrDistance(m.x, m.y, pi, pj) <= R_SQR;
  cellValidities[idxMinutia * NS * NS + threadIdx.y * NS + threadIdx.x] = validity;

  const int SIGMA_9S_SQR = 9 * SIGMA_S_SQR;
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
    cellValues[idx] = value;
  }

  int sumValidities = __syncthreads_count(validity);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int sumContribution = 0;
    for (int i = 0; i < N; ++i)
      sumContribution += contributed[i];

    cylinderValidities[idxMinutia] = sumContribution >= MIN_M &&
      sumValidities/(float)(NS*NS) >= MIN_VC);
  }
}

__host__
void buildTemplate(
    const vector<Minutia>& minutiae,
    const vector<char>& validArea,
    int width, int height,
    vector<char>& cylinderValidities,
    vector<char>& cellValues,
    vector<char>& cellValidities) {
  Minutia *devMinutiae;
  char *devArea;
  char *devCylinderValidities, *devCellValues, *devCellValidities;
  size_t devMinutiaeSize = minutiae.size() * sizeof(Minutia);
  size_t devAreaSize = width * height * sizeof(char);
  size_t devCylinderValiditiesSize = minutiae.size() * sizeof(char);
  size_t devCellValuesSize = minutiae.size() * NC * sizeof(char);
  size_t devCellValiditiesSize = minutiae.size() * NS * NS * sizeof(char);
  handleError(
    cudaMalloc(&devMinutiae, devMinutiaeSize));
  handleError(
    cudaMemcpy(devMinutiae, minutiae.data(), devMinutiaeSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devArea, devAreaSize));
  handleError(
    cudaMemcpy(devArea, validArea.data(), devAreaSize, cudaMemcpyHostToDevice));
  handleError(
    cudaMalloc(&devCylinderValidities, devCylinderValiditiesSize));
  handleError(
    cudaMalloc(&devCellValues, devCellValuesSize));
  handleError(
    cudaMalloc(&devCellValidities, devCellValiditiesSize));

  dim3 blockDim(NS, NS);
  int sharedSize = devMinutiaeSize + minutiae.size() * sizeof(char);
  buildCylinder<<<minutiae.size(), blockDim, sharedSize>>>(
    devMinutiae, devArea, width, height,
    devCylinderValidities, devCellValues, devCellValidities);

  cylinderValidities.resize(minutiae.size());
  cellValues.resize(minutiae.size() * NC);
  cellValidities.resize(minutiae.size() * NS * NS);
  handleError(
    cudaMemcpy(cylinderValidities.data(), devCylinderValidities, devCylinderValiditiesSize, cudaMemcpyDeviceToHost));
  handleError(
    cudaMemcpy(cellValues.data(), devCellValues, devCellValuesSize, cudaMemcpyDeviceToHost));
  handleError(
    cudaMemcpy(cellValidities.data(), devCellValidities, devCellValiditiesSize, cudaMemcpyDeviceToHost));

  cudaFree(devMinutiae);
  cudaFree(devArea);
  cudaFree(devCylinderValidities);
  cudaFree(devCellValues);
  cudaFree(devCellValidities);
}
