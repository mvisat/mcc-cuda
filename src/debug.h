#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cstdio>

#ifdef DEBUG
  #define debug(...) fprintf(stderr, __VA_ARGS__)
#else
  #define debug(...) do {} while(0)
#endif

#endif
