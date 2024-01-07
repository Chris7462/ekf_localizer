#pragma once

#include "kalman_filter/matrix.hpp"


namespace ekf_localizer
{

constexpr int StateSize = 6;

class State : public kalman::Vector<StateSize>
{
public:
  KALMAN_VECTOR(State, StateSize)
  enum : uint8_t
  {
    X,
    Y,
    THETA,
    NU,
    OMEGA,
    ALPHA
  };

  State(const double * data)
  : Vector<StateSize>()
  {
    for (int i = 0; i < StateSize; ++i) {
      (*this)[i] = data[i];
    }
  }

  inline double x() const {return (*this)[X];}
  inline double y() const {return (*this)[Y];}
  inline double theta() const {return (*this)[THETA];}
  inline double nu() const {return (*this)[NU];}
  inline double omega() const {return (*this)[OMEGA];}
  inline double alpha() const {return (*this)[ALPHA];}

  inline double & x() {return (*this)[X];}
  inline double & y() {return (*this)[Y];}
  inline double & theta() {return (*this)[THETA];}
  inline double & nu() {return (*this)[NU];}
  inline double & omega() {return (*this)[OMEGA];}
  inline double & alpha() {return (*this)[ALPHA];}
};

} // namespace ekf_localizer
