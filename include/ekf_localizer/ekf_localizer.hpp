#pragma once

// C++ header
#include <mutex>
#include <memory>
#include <string>
#include <vector>

// ros header
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2/LinearMath/Transform.hpp>

// local ros header
#include "av_msgs/msg/geo_plane_point.hpp"

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
  /**
   * @brief Constructor for EkfLocalizer node
   */
  EkfLocalizer();

  /**
   * @brief Destructor for EkfLocalizer node
   */
  ~EkfLocalizer() = default;

private:
  /**
   * @brief Initialize node parameters and EKF state/covariance, with validation
   * @return true if initialization successful, false otherwise
   */
  bool initialize_parameters();

  /**
   * @brief Initialize ROS2 publishers, subscribers, and timers with separate callback groups
   */
  void initialize_ros_components();

  /**
   * @brief Callback for incoming IMU messages; runs an EKF update immediately
   * @param msg Incoming IMU message
   */
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

  /**
   * @brief Callback for incoming GPS messages; runs an EKF update immediately
   * @param msg Incoming GPS message
   */
  void gps_callback(const av_msgs::msg::GeoPlanePoint::SharedPtr msg);

  /**
   * @brief Callback for incoming velocity messages; runs an EKF update immediately
   * @param msg Incoming velocity message
   */
  void vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

  /**
   * @brief Timer callback that runs the EKF predict step at a fixed rate
   */
  void predict_callback();

  /**
   * @brief Timer callback that publishes the current EKF state as the map->base_link TF
   */
  void publish_callback();

  // ROS2 topic / timing / QoS parameters
  std::string imu_input_topic_;
  std::string gps_input_topic_;
  std::string vel_input_topic_;
  double predict_freq_;
  double publish_freq_;
  int queue_size_;

  double alt_;  // for publish purpose
  double pitch_;  // for publish purpose
  double roll_; // for publish purpose

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<av_msgs::msg::GeoPlanePoint>::SharedPtr gps_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_sub_;
  rclcpp::TimerBase::SharedPtr predict_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // Callback groups for parallel execution
  rclcpp::CallbackGroup::SharedPtr sub_callback_group_;
  rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::mutex mtx_;

  SystemModel sys_;
  ImuMeasurementModel imu_model_;
  GpsMeasurementModel gps_model_;
  VelMeasurementModel vel_model_;
  EKF ekf_;
};

} // namespace ekf_localizer
