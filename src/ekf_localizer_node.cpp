// C++ header
#include <memory>

// ROS header
#include <rclcpp/executors/multi_threaded_executor.hpp>

// local header
#include "ekf_localizer/ekf_localizer.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Create the node
  auto node = std::make_shared<ekf_localizer::EkfLocalizer>();

  // EventsCBGExecutor: uses 10-15% less CPU than MultiThreadedExecutor,
  // supports multiple ROS time sources, and manages threading internally.
  rclcpp::executors::EventsCBGExecutor executor;

  // Add node to executor
  executor.add_node(node);

  RCLCPP_INFO(node->get_logger(), "Starting EKF Localizer with EventCBGExecutor");

  // Spin with multiple threads
  executor.spin();

  rclcpp::shutdown();

  return 0;
}
