#include <cmath>
#include "ekf_localizer/extended_kalman_filter.hpp"


namespace ekf_localizer
{

void EKF::wrapStateYaw()
{
  x_.theta() = wrap2pi(x_.theta());
}

double EKF::limitMeasurementYaw(double yaw) const
{
  double yaw_max = x_.theta() + M_PI;
  double yaw_min = x_.theta() - M_PI;
  while ((yaw > yaw_max) || (yaw < yaw_min)) {
    if (yaw > yaw_max) {
      yaw -= 2.0 * M_PI;
    } else if (yaw < yaw_min) {
      yaw += 2.0 * M_PI;
    }
  }
  return yaw;
}

double EKF::wrap2pi(const double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

} // namespace ekf_localizer
