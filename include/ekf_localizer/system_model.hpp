#pragma once

#include "kalman_filter/linearized_system_model.hpp"
#include "ekf_localizer/state.hpp"


namespace ekf_localizer
{

constexpr int ControlSize = 2;

class Control : public kalman::Vector<ControlSize>
{
public:
  KALMAN_VECTOR(Control, ControlSize)
  enum : uint8_t
  {
    OMEGA,
    ALPHA
  };

  inline double omega() const {return (*this)[OMEGA];}
  inline double alpha() const {return (*this)[ALPHA];}

  inline double & omega() {return (*this)[OMEGA];}
  inline double & alpha() {return (*this)[ALPHA];}
};

class SystemModel : public kalman::LinearizedSystemModel<State, Control, kalman::StandardBase>
{
public:
  SystemModel();
  ~SystemModel() = default;

  void set_dt(const double dt) {dt_ = dt;}

  State f(const State & x, const Control & u) const;
  void updateJacobians(const State & x, const Control & u);

private:
  double dt_{0.0};
};

} // namespace ekf_localizer
