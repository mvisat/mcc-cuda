#ifndef __BINARIZATION_CUH__
#define __BINARIZATION_CUH__

__host__
void devBinarizedTemplate(
  const int n,
  char *devCellValidities,
  char *devCellValues,
  unsigned int *devBinarizedValidities,
  unsigned int *devBinarizedValues);

#endif
