#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#include <cmath>

#define EPS 1e-9f
#define M_2PI (M_PI * 2)

#define MAX_MINUTIAE 256
#define BITS (8 * sizeof(unsigned int))

#define R 70
#define R_SQR (R * R)
#define NS 16
#define ND 6
#define NC (NS * NS * ND)
#define SIGMA_S (28.0f / 3)
#define SIGMA_S_SQR (SIGMA_S * SIGMA_S)
#define SIGMA_2S_SQR (2 * SIGMA_S_SQR)
#define SIGMA_3S 28
#define SIGMA_9S_SQR 784 // = (28/3)^2 * 9
#define SIGMA_D (2 * M_PI / 9)
#define DELTA_S (2 * R / (float)NS)
#define DELTA_D (2 * M_PI / ND)
#define DELTA_D_2 (DELTA_D / 2)
#define MU_PSI 0.01f
#define SIGMA 50
#define MIN_VC 0.75f
#define MIN_M 2
#define MIN_ME 0.6f
#define OMEGA 50

#define MU_P 20
#define TAU_P (2.0f/5)
#define MIN_NP 4
#define MAX_NP 12

#endif
