#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#include <cmath>

#define EPS 1e-9f
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279502884197169399375105820974f
#endif
#define M_2PI (M_PI * 2)

#define MAX_MINUTIAE 256
#define BITS (8 * sizeof(unsigned int))

#define R 70
#define NS 16
#define ND 6
#define SIGMA_S (28.0f / 3)
#define SIGMA_D (2 * M_PI / 9)
#define MU_PSI 0.01f
#define TAU_PSI 400
#define OMEGA 50
#define MIN_VC 0.75f
#define MIN_M 2
#define MIN_ME 0.6f
#define DELTA_THETA M_PI_2
#define MU_P 20
#define TAU_P (2.0f / 5)
#define MIN_NP 4
#define MAX_NP 12

#define DELTA_S (2.0f * R / NS)
#define DELTA_D (2 * M_PI / ND)

// Shorthand
#define R_SQR (R * R)
#define NC (NS * NS * ND)
#define SIGMA_S_SQR (SIGMA_S * SIGMA_S)
#define SIGMA_2S_SQR (2 * SIGMA_S_SQR)
#define SIGMA_3S 28
#define SIGMA_9S_SQR 784
#define DELTA_D_2 (DELTA_D / 2)
#define MIN_ME_CELLS 748 // floor(MIN_ME * 208 (base valid cells) * ND)

// 1 / (2 * SIGMA_S^2)
#define I_2_SIGMA_S_SQR 0.005739795918367346938775510204081632653061224489795918367f

// 1 / (SIGMA_S * sqrt(2*PI))
#define I_2_SIGMA_S_SQRT_PI 0.042743815757296358350708506421540914479556281910528713321f

// sqrt(PI/2) * (2*PI/9)
#define SQRT_PI_2_SIGMA_D 0.874978330317912208016146999692622474572871410155485214347f

// 1 / (sqrt(2) * (2*PI/9))
#define I_SQRT_2_SIGMA_D 1.012855855676744328249599089882583155497645918699623203767f

// 1 / (sqrt(2*PI) * (2*PI/9))
#define I_SQRT_2_PI_SIGMA_D 0.571442723408168728071869744411724181395485093540456316730f

#endif
