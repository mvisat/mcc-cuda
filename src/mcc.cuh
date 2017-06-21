#ifndef __MCC_CUH__
#define __MCC_CUH__

#include <vector>

#include "minutia.cuh"

class MCC {
public:
  MCC(const char *input);
  virtual ~MCC();

  bool load();
  bool build();
  void dispose();

  bool match(const char *target, float &similarity, std::vector<float> &matrix);

private:
  MCC();

  bool loaded, built;

  const char *input;
  int width, height, dpi, n;
  std::vector<Minutia> minutiae;

  Minutia *devMinutiae;
  char *devArea;
  char *devCylinderValidities;
  char *devCellValidities;
  char *devCellValues;
};

#endif
