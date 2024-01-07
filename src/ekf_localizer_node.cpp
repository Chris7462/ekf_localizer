#include "ekf_localizer/ekf_localizer.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ekf_localizer::EkfLocalizer>());
  rclcpp::shutdown();

  return 0;
}
