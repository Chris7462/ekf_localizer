#include "ekf_localizer/vel_measurement_model.hpp"


namespace ekf_localizer
{

VelMeasurementModel::VelMeasurementModel()
{
  chisq_threshold_ = 3.841458821; // chi-square dist, 0.95 quantile with df = 1
  H_.setZero();
  V_.setIdentity();
}

VelMeasurement VelMeasurementModel::h(const State & x) const
{
  VelMeasurement measurement;
  measurement.nu() = x.nu();

  return measurement;
}

void VelMeasurementModel::updateJacobians(const State & x)
{
  (void)x;  // not using state. Just put this line to silence compiler warning.

  // H = dh/dx (Jacobian of measurement transition w.r.t the state)
  H_.setZero();
  H_(VelMeasurement::NU, State::NU) = 1.0;

  // V = dh/ddelta (Jacobian of measurement trasition w.r.t the noise)
  V_.setIdentity();
}

} // namespace ekf_localizer
