#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#include <cmath>

#define EPS 1e-9f
__constant__ const float M_2PI = M_PI * 2;

#define MAX_MINUTIAE 256

#define R 70
__constant__ const int R_SQR = R * R;
#define NS 16
#define ND 6
__constant__ const int NC = NS * NS * ND;
__constant__ const float SIGMA_S = 28.0f / 3;
__constant__ const float SIGMA_S_SQR = SIGMA_S * SIGMA_S;
__constant__ const float SIGMA_2S_SQR = 2 * SIGMA_S_SQR;
#define SIGMA_3S 28
#define SIGMA_9S_SQR 784 // = (28/3)^2 * 9
__constant__ const float SIGMA_D = 2 * M_PI / 9;
__constant__ const float DELTA_S = (float)(R * 2) / NS;
__constant__ const float DELTA_D = M_PI * 2 / ND;
__constant__ const float DELTA_D_2 = DELTA_D / 2;
#define MU_PSI 0.01f
#define SIGMA 50
#define MIN_VC 0.75f
#define MIN_M 2
#define MIN_ME 0.6f
#define OMEGA 50

#endif
