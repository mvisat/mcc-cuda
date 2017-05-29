#ifndef __MINUTIA_CUH__
#define __MINUTIA_CUH__

#include <ostream>

struct __align__(16) Minutia {
public:
  int x, y;
  float theta;

  __host__ __device__
  Minutia(int x, int y, float theta): x(x), y(y), theta(theta) {};

  __host__ __device__
  Minutia(const Minutia& other): x(other.x), y(other.y), theta(other.theta) {};

  __host__ __device__
  Minutia& operator=(const Minutia& other) {
    if (this != &other) {
      x = other.x;
      y = other.y;
      theta = other.theta;
    }
    return *this;
  };

  __host__
  friend std::ostream &operator<<(std::ostream &stream, Minutia &m) {
    stream << m.x << ' ' << m.y << ' ' << m.theta;
    return stream;
  };

private:
  Minutia();
};

#endif
