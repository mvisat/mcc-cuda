#ifndef __MCC_CUH__
#define __MCC_CUH__

#include <vector>

#include "minutia.cuh"

class MCC {
public:
  MCC();
  MCC(const char *input, bool autoLoad = true);
  virtual ~MCC();

  bool load();
  bool load(const char *input);
  bool build();

  bool match(const char *target,
    float &similarity, int &n, int &m, std::vector<float> &matrix);
  void matchMany(const std::vector<std::string> &targets, std::vector<float> &values);

private:
  void initialize();
  void allocate();
  void dispose();

  const char *input;
  int width, height, dpi, n;
  std::vector<Minutia> minutiae;

  Minutia *devMinutiae;
  char *devArea;
  char *devCylinderValidities;
  char *devCellValidities;
  char *devCellValues;
  unsigned int *devBinarizedValidities;
  unsigned int *devBinarizedValues;
  float *devMatrix;
};

#endif
