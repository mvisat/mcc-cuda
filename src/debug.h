#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cstdio>

#ifdef DEBUG
  #define debug(...) fprintf(stderr, __VA_ARGS__)
  #define devDebug(...) printf(__VA_ARGS__)
#else
  #define debug(...) do {} while(0)
  #define devDebug(...) do {} while(0)
#endif

#endif
