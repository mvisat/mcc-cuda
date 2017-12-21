#ifndef __SORT_CUH__
#define __SORT_CUH__

template<class T>
__global__
void bitonicSortStep(T *devArray, int j, int k) {
  auto i =  threadIdx.x + blockDim.x * blockIdx.x;
  auto ixj = i ^ j;
  if (ixj <= i) return;

  if ((i&k) == 0) {
    if (devArray[i] < devArray[ixj]) {
      auto temp = devArray[i];
      devArray[i] = devArray[ixj];
      devArray[ixj] = temp;
    }
  } else {
    if (devArray[i] > devArray[ixj]) {
      auto temp = devArray[i];
      devArray[i] = devArray[ixj];
      devArray[ixj] = temp;
    }
  }
}

template<class T>
__host__
void devBitonicSort(T *devArray, int N) {
  int numThreads = 1024;
  dim3 blocks(ceilMod(N, numThreads));
  dim3 threads(numThreads);

  for (int k = 2; k <= N; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1)
      bitonicSortStep<T><<<blocks, threads>>>(devArray, j, k);
}

#endif
