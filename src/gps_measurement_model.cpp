#include "ekf_localizer/gps_measurement_model.hpp"


namespace ekf_localizer
{

GpsMeasurementModel::GpsMeasurementModel()
{
  chisq_threshold_ = 5.991464547; // chi-square dist, 0.95 quantile with df = 2
  H_.setZero();
  V_.setIdentity();
}

GpsMeasurement GpsMeasurementModel::h(const State & x) const
{
  GpsMeasurement measurement;
  measurement.x() = x.x();
  measurement.y() = x.y();

  return measurement;
}

void GpsMeasurementModel::updateJacobians(const State & x)
{
  (void)x;  // not using state. Just put this line to silence compiler warning.

  // H = dh/dx (Jacobian of measurement transition w.r.t the state)
  H_.setZero();
  H_(GpsMeasurement::X, State::X) = 1.0;
  H_(GpsMeasurement::Y, State::Y) = 1.0;

  // V = dh/ddelta (Jacobian of measurement trasition w.r.t the noise)
  V_.setIdentity();
}

} // namespace ekf_localizer
