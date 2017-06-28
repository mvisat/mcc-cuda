#include "io.cuh"
#include "constants.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace std;

bool loadMinutiaeFromFile(
    const char *input,
    int &width, int &height, int &dpi, int &n,
    vector<Minutia> &minutiae) {
  ifstream istream;
  istream.exceptions(ifstream::failbit | ifstream::badbit);
  try {
    istream.open(input);
    istream >> width >> height >> dpi >> n;
    for (int i = 0; i < n; ++i) {
      int x, y;
      float theta;
      istream >> x >> y >> theta;
      minutiae.emplace_back(x, y, theta);
    }
  } catch (const ifstream::failure &e) {
    cerr << "error when reading minutiae file: " << input << endl;
    cerr << e.what() << endl;
    return false;
  }
  istream.close();
  return true;
}

bool loadTemplateFromFile(
    const char *input,
    int &width, int &height, int &dpi, int &n,
    vector<Minutia> &minutiae,
    int &m,
    vector<char> &cylinderValidities,
    vector<char> &cellValidities,
    vector<char> &cellValues) {
  ifstream istream;
  istream.exceptions(ifstream::failbit | ifstream::badbit);
  try {
    istream.open(input);
    istream >> width >> height >> dpi >> n;
    for (int i = 0; i < n; ++i) {
      int x, y;
      float theta;
      istream >> x >> y >> theta;
      minutiae.emplace_back(x, y, theta);
    }
    istream >> m;
    cylinderValidities.resize(m);
    cellValidities.reserve(m * NC);
    cellValues.reserve(m * NC);
    for (int l = 0; l < m; ++l) {
      string s;
      istream >> s;
      cylinderValidities[l] = s.compare("True") == 0 ? 1 : 0;
      if (!cylinderValidities[l]) {
        for (int i = 0; i < NC; ++i) {
          cellValidities.push_back(0);
          cellValues.push_back(0);
        }
        continue;
      }
      for (int i = 0; i < NS; ++i) {
        for (int j = 0; j < NS; ++j) {
          int validity;
          istream >> validity;
          for (int k = 0; k < ND; ++k) {
            cellValidities.push_back(validity);
          }
        }
      }
      for (int i = 0; i < NS; ++i) {
        for (int j = 0; j < NS; ++j) {
          for (int k = 0; k < ND; ++k) {
            int value;
            istream >> value;
            cellValues.push_back(value);
          }
        }
      }
    }
  } catch (const ifstream::failure &e) {
    cerr << "error when reading template file: " << input << endl;
    cerr << e.what() << endl;
    return false;
  }
  istream.close();
  return true;
}

bool saveTemplateToFile(
    const char *output,
    int width, int height, int dpi, int n,
    const vector<Minutia> &minutiae,
    int m,
    const vector<char> &cylinderValidities,
    const vector<char> &cellValidities,
    const vector<char> &cellValues) {
  ofstream ostream(output);
  ostream.exceptions(ofstream::failbit | ofstream::badbit);
  try {
    ostream.precision(numeric_limits<double>::max_digits10);
    ostream << width << endl;
    ostream << height << endl;
    ostream << dpi << endl;
    ostream << n << endl;
    for (auto &m: minutiae)
    ostream << m.x << ' ' << m.y << ' ' << m.theta << endl;
    ostream << n << endl;
    for (int i = 0; i < n; ++i) {
      if (cylinderValidities[i]) {
        ostream << "True ";
        for (int j = 0; j < NS; ++j) {
          for (int k = 0; k < NS; ++k) {
            ostream << (cellValidities[i*NC + j*NS*ND + k*ND] ? "1 " : "0 ");
          }
        }
        for (int j = 0; j < NC; ++j) {
          ostream << (cellValues[i*NC + j] ? '1' : '0');
          if (j != NC-1) ostream << ' ';
        }
      } else {
        ostream << "False";
      }
      ostream << endl;
    }
  } catch (const ofstream::failure &e) {
    cerr << "error when writing template file: " << output << endl;
    cerr << e.what() << endl;
    return false;
  }
  ostream.close();
  return true;
}

bool saveSimilarityToFile(
    const char *output,
    const int n, const int m,
    const vector<float> &matrix) {
  ofstream ostream;
  ostream.exceptions(ofstream::failbit | ofstream::badbit);
  try {
    ostream.open(output);
    ostream.precision(numeric_limits<double>::max_digits10);
    ostream << n << endl;
    ostream << m << endl;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        ostream << matrix[i*m+j] << (j == m-1 ? '\n' : ' ');
      }
    }
  } catch (const ofstream::failure &e) {
    cerr << "error when writing similarity file: " << output << endl;
    cerr << e.what() << endl;
    return false;
  }
  ostream.close();
  return true;
}
