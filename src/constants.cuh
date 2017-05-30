#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#include <cmath>

#define MAX_MINUTIAE 256
#define ROWS 100
#define COLS 100

#define EPS 1e-9
__constant__ const int R = 70;
__constant__ const int R2 = R * R;
__constant__ const int NS = 8;
__constant__ const int ND = 6;
__constant__ const int NC = NS * NS * ND;
__constant__ const float SIGMA_S = 28.0f / 3;
__constant__ const float SIGMA_D = 2.0f * M_PI / 9;
__constant__ const float DELTA_S = R * 2.0f / NS;
__constant__ const float DELTA_D = M_PI * 2 / ND;
__constant__ const float MU_PSI = 0.01f;
__constant__ const int SIGMA = 50;
__constant__ const float MIN_VC = 0.75f;
__constant__ const int MIN_M = 2;
__constant__ const float MIN_ME = 0.6f;
#define OMEGA 50

#endif
