#pragma once

#include "kalman_filter/linearized_measurement_model.hpp"
#include "ekf_localizer/state.hpp"


namespace ekf_localizer
{

constexpr int ImuMeasurementSize = 3;

class ImuMeasurement : public kalman::Vector<ImuMeasurementSize>
{
public:
  KALMAN_VECTOR(ImuMeasurement, ImuMeasurementSize)
  enum : uint8_t
  {
    THETA,
    OMEGA,
    ALPHA
  };

  inline double theta() const {return (*this)[THETA];}
  inline double omega() const {return (*this)[OMEGA];}
  inline double alpha() const {return (*this)[ALPHA];}

  inline double & theta() {return (*this)[THETA];}
  inline double & omega() {return (*this)[OMEGA];}
  inline double & alpha() {return (*this)[ALPHA];}
};

class ImuMeasurementModel
  : public kalman::LinearizedMeasurementModel<State, ImuMeasurement, kalman::StandardBase>
{
public:
  ImuMeasurementModel();
  ~ImuMeasurementModel() = default;

  ImuMeasurement h(const State & x) const;
  void updateJacobians(const State & x);
};

} // namespace ekf_localizer
