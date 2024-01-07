#include "ekf_localizer/imu_measurement_model.hpp"


namespace ekf_localizer
{

ImuMeasurementModel::ImuMeasurementModel()
{
  chisq_threshold_ = 7.814727903; // chi-square dist, 0.95 quantile with df = 3
  H_.setZero();
  V_.setIdentity();
}

ImuMeasurement ImuMeasurementModel::h(const State & x) const
{
  ImuMeasurement measurement;
  measurement.theta() = x.theta();
  measurement.omega() = x.omega();
  measurement.alpha() = x.alpha();

  return measurement;
}

void ImuMeasurementModel::updateJacobians(const State & x)
{
  (void)x;  // not using state. Just put this line to silence compiler warning.

  // H = dh/dx (Jacobian of measurement transition w.r.t the state)
  H_.setZero();
  H_(ImuMeasurement::THETA, State::THETA) = 1.0;
  H_(ImuMeasurement::OMEGA, State::OMEGA) = 1.0;
  H_(ImuMeasurement::ALPHA, State::ALPHA) = 1.0;

  // V = dh/ddelta (Jacobian of measurement trasition w.r.t the noise)
  V_.setIdentity();
}

} // namespace ekf_localizer
