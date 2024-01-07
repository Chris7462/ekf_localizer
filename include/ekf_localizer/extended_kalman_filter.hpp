#pragma once

#include "kalman_filter/extended_kalman_filter.hpp"
#include "ekf_localizer/state.hpp"

namespace ekf_localizer
{

class EKF : public kalman::ExtendedKalmanFilter<State>
{
public:
  void wrapStateYaw();
  double limitMeasurementYaw(double yaw) const;
  double wrap2pi(const double angle);
};

} // namespace ekf_localizer
