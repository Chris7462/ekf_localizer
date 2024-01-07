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
  SystemModel(const double dt);
  ~SystemModel() = default;

  State f(const State & x, const Control & u) const;
  void updateJacobians(const State & x, const Control & u);

private:
  double dt_;
};

} // namespace ekf_localizer
