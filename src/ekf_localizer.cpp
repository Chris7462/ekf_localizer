// C++ header
#include <chrono>

// ROS header
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// local header
#include "ekf_localizer/ekf_localizer.hpp"


namespace ekf_localizer
{

EkfLocalizer::EkfLocalizer()
: Node("ekf_localizer_node"), freq_{40.0}, alt_{0.0}, pitch_{0.0}, roll_{0.0},
  odom_base_link_trans_(), imu_buff_(), gps_buff_(), vel_buff_(), sys_(1.0 / freq_),
  imu_model_(), gps_model_(), vel_model_(), ekf_()
{
  rclcpp::QoS qos(10);

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "kitti/oxts/imu_rotated", qos,
    std::bind(&EkfLocalizer::imu_callback, this, std::placeholders::_1));

  gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
    "kitti/oxts/gps_shifted", qos,
    std::bind(&EkfLocalizer::gps_callback, this, std::placeholders::_1));

  vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    "kitti/oxts/gps/vel", qos,
    std::bind(&EkfLocalizer::vel_callback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1.0 / freq_ * 1000)),
    std::bind(&EkfLocalizer::run_ekf, this));

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  odom_base_link_trans_.setIdentity();

  // Set system model covariance
  double eps_x = declare_parameter("eps.x", 0.0);
  double eps_y = declare_parameter("eps.y", 0.0);
  double eps_theta = declare_parameter("eps.theta", 0.0);
  double eps_nu = declare_parameter("eps.nu", 0.0);
  double eps_omega = declare_parameter("eps.omega", 0.0);
  double eps_alpha = declare_parameter("eps.alpha", 0.0);
  kalman::Covariance<State> Q = kalman::Covariance<State>::Zero();
  Q.diagonal() << eps_x, eps_y, eps_theta, eps_nu, eps_omega, eps_alpha;
  sys_.setCovariance(Q);

  // Set IMU measurement covariance
  double tau_theta = declare_parameter("tau.theta", 0.0);
  double tau_omega = declare_parameter("tau.omega", 0.0);
  double tau_alpha = declare_parameter("tau.alpha", 0.0);
  kalman::Covariance<ImuMeasurement> RI = kalman::Covariance<ImuMeasurement>::Zero();
  RI.diagonal() << tau_theta, tau_omega, tau_alpha;
  imu_model_.setCovariance(RI);

  // Set GPS measurement covariance
  double tau_x = declare_parameter("tau.x", 0.0);
  double tau_y = declare_parameter("tau.y", 0.0);
  kalman::Covariance<GpsMeasurement> RG = kalman::Covariance<GpsMeasurement>::Zero();
  RG.diagonal() << tau_x, tau_y;
  gps_model_.setCovariance(RG);

  // Set Vel measurement covariance
  double tau_nu = declare_parameter("tau.nu", 0.0);
  kalman::Covariance<VelMeasurement> RV = kalman::Covariance<VelMeasurement>::Zero();
  RV.diagonal() << tau_nu;
  vel_model_.setCovariance(RV);

  // Init ekf state
  std::vector<double> vec_x = declare_parameter("init.x", std::vector<double>());
  State init_x(vec_x.data());
  ekf_.init(init_x);

  // Init ekf covariance
  std::vector<double> vec_P = declare_parameter("init.P", std::vector<double>());
  kalman::Covariance<State> init_P(vec_P.data());
  ekf_.setCovariance(init_P);
}

void EkfLocalizer::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  imu_buff_.push(msg);
}

void EkfLocalizer::gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  gps_buff_.push(msg);
}

void EkfLocalizer::vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  vel_buff_.push(msg);
}

void EkfLocalizer::run_ekf()
{
  rclcpp::Time current_time = rclcpp::Node::now();
  auto s = ekf_.getState();

  // Run predict
  ekf_.predict(sys_);
  ekf_.wrapStateYaw();

  // Run imu update
  if (!imu_buff_.empty()) {
    mtx_.lock();
    if ((current_time - rclcpp::Time(imu_buff_.front()->header.stamp)).seconds() > 0.1) {  // time sync has problem
      imu_buff_.pop();
      RCLCPP_WARN(this->get_logger(), "Timestamp unaligned, please check your IMU data.");
      mtx_.unlock();
    } else {
      auto msg = imu_buff_.front();
      imu_buff_.pop();
      mtx_.unlock();

      double yaw;
      tf2::getEulerYPR(msg->orientation, yaw, pitch_, roll_);

      ImuMeasurement z;
      z.theta() = ekf_.limitMeasurementYaw(yaw);
      z.omega() = msg->angular_velocity.z;
      z.alpha() = msg->linear_acceleration.x;

      // check imu update successful?
      if (ekf_.update(imu_model_, z)) {
        ekf_.wrapStateYaw();

        // publish odom to base link TF
        tf2::Vector3 t_current(s.x(), s.y(), alt_);
        tf2::Quaternion q_current;
        q_current.setRPY(roll_, pitch_, s.theta());
        q_current.normalize();

        geometry_msgs::msg::TransformStamped odom_base_link_tf;
        odom_base_link_tf.header.stamp = rclcpp::Node::now();
        odom_base_link_tf.header.frame_id = "odom";
        odom_base_link_tf.child_frame_id = "base_link";
        odom_base_link_tf.transform.translation = tf2::toMsg(t_current);
        odom_base_link_tf.transform.rotation = tf2::toMsg(q_current);

        // send the transformation
        tf_broadcaster_->sendTransform(odom_base_link_tf);

        // save the odom->base transform
        tf2::fromMsg(odom_base_link_tf.transform, odom_base_link_trans_);
      } else {
        RCLCPP_INFO(
          get_logger(), "Measurement IMU is over the threshold. Discard this measurement.");
      }
    }
  }

  // Run gps update
  if (!gps_buff_.empty()) {
    mtx_.lock();
    if ((current_time - rclcpp::Time(gps_buff_.front()->header.stamp)).seconds() > 0.1) {  // time sync has problem
      gps_buff_.pop();
      RCLCPP_WARN(get_logger(), "Timestamp unaligned, please check your GPS data.");
      mtx_.unlock();
    } else {
      auto msg = gps_buff_.front();
      gps_buff_.pop();
      mtx_.unlock();

      // gps measurement
      GpsMeasurement z;
      z.x() = msg->latitude;
      z.y() = msg->longitude;
      alt_ = msg->altitude;

      // use the covariance that Gps provided.
      kalman::Covariance<GpsMeasurement> R = kalman::Covariance<GpsMeasurement>::Zero();
      R.diagonal() << msg->position_covariance.at(0), msg->position_covariance.at(4);
      gps_model_.setCovariance(R);

      // check GPS update successful?
      if (ekf_.update(gps_model_, z)) {
        ekf_.wrapStateYaw();
      } else {
        RCLCPP_INFO(
          get_logger(), "Measurement GPS is over the threshold. Discard this measurement.");
      }
    }
  }

  // publish map to odom TF
  tf2::Vector3 t_current(s.x(), s.y(), alt_);
  tf2::Quaternion q_current;
  q_current.setRPY(roll_, pitch_, s.theta());
  q_current.normalize();

  tf2::Transform map_base_link_trans(q_current, t_current);
  tf2::Transform map_odom_trans;
  map_odom_trans.mult(map_base_link_trans, odom_base_link_trans_.inverse());

  geometry_msgs::msg::TransformStamped map_odom_tf;
  map_odom_tf.header.stamp = rclcpp::Node::now();
  map_odom_tf.header.frame_id = "map";
  map_odom_tf.child_frame_id = "odom";
  map_odom_tf.transform = tf2::toMsg(map_odom_trans);

  // send the transformation
  tf_broadcaster_->sendTransform(map_odom_tf);

  // Run velocity update
  if (!vel_buff_.empty()) {
    mtx_.lock();
    if ((current_time - rclcpp::Time(vel_buff_.front()->header.stamp)).seconds() > 0.1) {  // time sync has problem
      vel_buff_.pop();
      RCLCPP_WARN(this->get_logger(), "Timestamp unaligned, please check your Velocity data.");
      mtx_.unlock();
    } else {
      auto msg = vel_buff_.front();
      vel_buff_.pop();
      mtx_.unlock();

      VelMeasurement z;
      z.nu() = msg->twist.linear.x;

      // check velocity update successful?
      if (ekf_.update(vel_model_, z)) {
        ekf_.wrapStateYaw();

        // publish odom to base link TF
        tf2::Vector3 t_current(s.x(), s.y(), alt_);
        tf2::Quaternion q_current;
        q_current.setRPY(roll_, pitch_, s.theta());
        q_current.normalize();

        geometry_msgs::msg::TransformStamped odom_base_link_tf;
        odom_base_link_tf.header.stamp = rclcpp::Node::now();
        odom_base_link_tf.header.frame_id = "odom";
        odom_base_link_tf.child_frame_id = "base_link";
        odom_base_link_tf.transform.translation = tf2::toMsg(t_current);
        odom_base_link_tf.transform.rotation = tf2::toMsg(q_current);

        // send the transformation
        tf_broadcaster_->sendTransform(odom_base_link_tf);

        // save the odom->base transform
        tf2::fromMsg(odom_base_link_tf.transform, odom_base_link_trans_);
      } else {
        RCLCPP_INFO(
          get_logger(), "Measurement Velocity is over the threshold. Discard this measurement.");
      }
    }
  }
}

}  // namespace ekf_localizer
