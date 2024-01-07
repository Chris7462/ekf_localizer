#include "ekf_localizer/state.hpp"
#include "ekf_localizer/system_model.hpp"


namespace ekf_localizer
{

SystemModel::SystemModel(const double dt)
: dt_{dt}
{
  F_.setIdentity();
  W_.setZero();
}

State SystemModel::f(const State & x, const Control & u) const
{
  (void)u;  // not using control here. Just put this line to silence compile warning.

  State state;
  // Predicted state vector after transition
  double dt2 = dt_ * dt_;
  double dl = x.nu() * dt_ + 0.5 * x.alpha() * dt2;
  double dtheta = x.theta() + 0.5 * x.omega() * dt_;

  state.x() = x.x() + dl * std::cos(dtheta);
  state.y() = x.y() + dl * std::sin(dtheta);
  state.theta() = x.theta() + x.omega() * dt_;
  state.nu() = x.nu() + x.alpha() * dt_;
  state.omega() = x.omega();
  state.alpha() = x.alpha();

  return state;
}

void SystemModel::updateJacobians(const State & x, const Control & u)
{
  (void)u;  // not using control here. Just put this line to silence compile warning.

  double dt2 = dt_ * dt_;
  double dl = x.nu() * dt_ + 0.5 * x.alpha() * dt2;
  double dtheta = x.theta() + 0.5 * x.omega() * dt_;
  double sin_dtheta = std::sin(dtheta);
  double cos_dtheta = std::cos(dtheta);

  // F = df/dx (Jacobian of state transition w.r.t the state)
  F_.setIdentity();

  // partial derivative of f_x w.r.t theta, nu, omega, alpha
  F_(State::X, State::THETA) = -dl * sin_dtheta;
  F_(State::X, State::NU) = dt_ * cos_dtheta;
  F_(State::X, State::OMEGA) = -0.5 * dt_ * dl * sin_dtheta;
  F_(State::X, State::ALPHA) = 0.5 * dt2 * cos_dtheta;

  // partial derivative of f_y w.r.t theta, nu, omega, alpha
  F_(State::Y, State::THETA) = dl * cos_dtheta;
  F_(State::Y, State::NU) = dt_ * sin_dtheta;
  F_(State::Y, State::OMEGA) = 0.5 * dt_ * dl * cos_dtheta;
  F_(State::Y, State::ALPHA) = 0.5 * dt2 * sin_dtheta;

  // partial derivative of f_theta w.r.t omega
  F_(State::THETA, State::OMEGA) = dt_;

  // partial derivative of f_nu w.r.t alpah
  F_(State::NU, State::ALPHA) = dt_;

  // W = df/depsilon  (Jacobian of state transition w.r.t the noise)
  // Assume the noise only coming from omega and alpha
  W_.setZero();

  // partial derivative of f_x w.r.t omega, alpha
  W_(State::X, State::OMEGA) = -0.5 * dt_ * dl * sin_dtheta;
  W_(State::X, State::ALPHA) = 0.5 * dt2 * cos_dtheta;

  // partial derivative of f_y w.r.t omega, alpha
  W_(State::Y, State::OMEGA) = 0.5 * dt_ * dl * cos_dtheta;
  W_(State::Y, State::ALPHA) = 0.5 * dt2 * sin_dtheta;

  // partial derivative of f_theta w.r.t omega
  W_(State::THETA, State::OMEGA) = dt_;

  // parital derivative of f_nu w.r.t alpah
  W_(State::NU, State::ALPHA) = dt_;

  // parital derivative of f_omega w.r.t omega
  W_(State::OMEGA, State::OMEGA) = 1.0;

  // parital derivative of f_alpha w.r.t alpha
  W_(State::ALPHA, State::ALPHA) = 1.0;
}

} // namespace ekf_localizer
