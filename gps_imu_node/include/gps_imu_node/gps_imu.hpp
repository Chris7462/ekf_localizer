#pragma once

// ros header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_broadcaster.h>

// geographicLib header
#include <GeographicLib/LocalCartesian.hpp>


namespace gps_imu_node
{

class GpsImuNode : public rclcpp::Node
{
public:
  GpsImuNode();
  ~GpsImuNode() = default;

private:
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  GeographicLib::LocalCartesian geo_converter_;
  bool gps_init_;

  message_filters::Subscriber<sensor_msgs::msg::Imu> sub_imu_;
  message_filters::Subscriber<sensor_msgs::msg::NavSatFix> sub_gps_;

  using policy_t = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Imu, sensor_msgs::msg::NavSatFix>;

  message_filters::Synchronizer<policy_t> sync_;

  void sync_callback(
    const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
    const sensor_msgs::msg::NavSatFix::ConstSharedPtr gps_msg);
};

}
