#pragma once

#include "kalman_filter/linearized_measurement_model.hpp"
#include "ekf_localizer/state.hpp"


namespace ekf_localizer
{

constexpr int GpsMeasurementSize = 2;

class GpsMeasurement : public kalman::Vector<GpsMeasurementSize>
{
public:
  KALMAN_VECTOR(GpsMeasurement, GpsMeasurementSize)
  enum : uint8_t
  {
    X,
    Y
  };

  inline double x() const {return (*this)[X];}
  inline double y() const {return (*this)[Y];}

  inline double & x() {return (*this)[X];}
  inline double & y() {return (*this)[Y];}
};

class GpsMeasurementModel
  : public kalman::LinearizedMeasurementModel<State, GpsMeasurement, kalman::StandardBase>
{
public:
  GpsMeasurementModel();
  ~GpsMeasurementModel() = default;

  GpsMeasurement h(const State & x) const;
  void updateJacobians(const State & x);
};

} // namespace ekf_localizer
