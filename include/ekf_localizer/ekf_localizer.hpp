#pragma once

// C++ header
#include <queue>
#include <mutex>
#include <memory>

// ros header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Transform.h>

// local header
#include "ekf_localizer/system_model.hpp"
#include "ekf_localizer/gps_measurement_model.hpp"
#include "ekf_localizer/imu_measurement_model.hpp"
#include "ekf_localizer/vel_measurement_model.hpp"
#include "ekf_localizer/extended_kalman_filter.hpp"


namespace ekf_localizer
{

class EkfLocalizer : public rclcpp::Node
{
public:
  EkfLocalizer();
  ~EkfLocalizer() = default;

private:
  double freq_;

  double alt_;  // for publish purpose
  double pitch_;  // for publish purpose
  double roll_; // for publish purpose

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
  void vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  tf2::Transform odom_base_link_trans_;

  std::queue<sensor_msgs::msg::Imu::SharedPtr> imu_buff_;
  std::queue<sensor_msgs::msg::NavSatFix::SharedPtr> gps_buff_;
  std::queue<geometry_msgs::msg::TwistStamped::SharedPtr> vel_buff_;

  std::mutex mtx_;

  void run_ekf();

  SystemModel sys_;
  ImuMeasurementModel imu_model_;
  GpsMeasurementModel gps_model_;
  VelMeasurementModel vel_model_;
  EKF ekf_;
};

} // namespace ekf_localizer
