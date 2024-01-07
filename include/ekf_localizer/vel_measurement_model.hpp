#pragma once

#include "kalman_filter/linearized_measurement_model.hpp"
#include "ekf_localizer/state.hpp"


namespace ekf_localizer
{

constexpr int VelMeasurementSize = 1;

class VelMeasurement : public kalman::Vector<VelMeasurementSize>
{
public:
  KALMAN_VECTOR(VelMeasurement, VelMeasurementSize)
  enum : uint8_t
  {
    NU
  };

  inline double nu() const {return (*this)[NU];}

  inline double & nu() {return (*this)[NU];}
};

class VelMeasurementModel
  : public kalman::LinearizedMeasurementModel<State, VelMeasurement, kalman::StandardBase>
{
public:
  VelMeasurementModel();
  ~VelMeasurementModel() = default;

  VelMeasurement h(const State & x) const;
  void updateJacobians(const State & x);
};

} // namespace ekf_localizer
