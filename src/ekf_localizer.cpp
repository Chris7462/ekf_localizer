// C++ header
#include <chrono>
#include <exception>

// ROS header
#include <tf2/utils.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// local header
#include "ekf_localizer/ekf_localizer.hpp"


namespace ekf_localizer
{

EkfLocalizer::EkfLocalizer()
: Node("ekf_localizer_node"), alt_{0.0}, pitch_{0.0}, roll_{0.0},
  sys_(), imu_model_(), gps_model_(), vel_model_(), ekf_()
{
  // Initialize and validate ROS2 parameters, and configure the EKF from them
  if (!initialize_parameters()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize parameters");
    rclcpp::shutdown();
    return;
  }

  // Initialize ROS2 publishers, subscribers, timers, and TF broadcaster
  initialize_ros_components();

  RCLCPP_INFO(get_logger(),
    "EKF localizer node initialized successfully "
    "(fixed-rate predict, event-driven update, independent publish rate)");
}

bool EkfLocalizer::initialize_parameters()
{
  try {
    // ROS2 topic / timing / QoS parameters
    imu_input_topic_ = declare_parameter("imu_input_topic",
      std::string("kitti/vehicle/imu_local"));
    gps_input_topic_ = declare_parameter("gps_input_topic",
      std::string("kitti/vehicle/gps_local"));
    vel_input_topic_ = declare_parameter("vel_input_topic",
      std::string("kitti/vehicle/velocity"));

    predict_freq_ = declare_parameter<double>("predict_frequency", 40.0);
    if (predict_freq_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid predict frequency: %.2f Hz", predict_freq_);
      return false;
    }

    publish_freq_ = declare_parameter<double>("publish_frequency", 40.0);
    if (publish_freq_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid publish frequency: %.2f Hz", publish_freq_);
      return false;
    }

    queue_size_ = declare_parameter<int>("queue_size", 10);
    if (queue_size_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid queue size: %d (should be > 0)", queue_size_);
      return false;
    }

    // Configure system model time step from the fixed processing frequency
    sys_.set_dt(1.0 / predict_freq_);

    // System model covariance
    double eps_x = declare_parameter("eps.x", 0.0);
    double eps_y = declare_parameter("eps.y", 0.0);
    double eps_theta = declare_parameter("eps.theta", 0.0);
    double eps_nu = declare_parameter("eps.nu", 0.0);
    double eps_omega = declare_parameter("eps.omega", 0.0);
    double eps_alpha = declare_parameter("eps.alpha", 0.0);
    kalman::Covariance<State> Q = kalman::Covariance<State>::Zero();
    Q.diagonal() << eps_x, eps_y, eps_theta, eps_nu, eps_omega, eps_alpha;
    sys_.setCovariance(Q);

    // IMU measurement covariance
    double tau_theta = declare_parameter("tau.theta", 0.0);
    double tau_omega = declare_parameter("tau.omega", 0.0);
    double tau_alpha = declare_parameter("tau.alpha", 0.0);
    kalman::Covariance<ImuMeasurement> RI = kalman::Covariance<ImuMeasurement>::Zero();
    RI.diagonal() << tau_theta, tau_omega, tau_alpha;
    imu_model_.setCovariance(RI);

    // GPS measurement covariance
    double tau_x = declare_parameter("tau.x", 0.0);
    double tau_y = declare_parameter("tau.y", 0.0);
    kalman::Covariance<GpsMeasurement> RG = kalman::Covariance<GpsMeasurement>::Zero();
    RG.diagonal() << tau_x, tau_y;
    gps_model_.setCovariance(RG);

    // Vel measurement covariance
    double tau_nu = declare_parameter("tau.nu", 0.0);
    kalman::Covariance<VelMeasurement> RV = kalman::Covariance<VelMeasurement>::Zero();
    RV.diagonal() << tau_nu;
    vel_model_.setCovariance(RV);

    // Initial state
    std::vector<double> init_x_vec = declare_parameter("init.x", std::vector<double>());
    if (init_x_vec.size() != 6) {
      RCLCPP_ERROR(get_logger(),
        "Invalid init.x size: %ld (expected 6)", init_x_vec.size());
      return false;
    }
    ekf_.init(State(init_x_vec.data()));

    // Initial covariance
    std::vector<double> init_P_vec = declare_parameter("init.P", std::vector<double>());
    if (init_P_vec.size() != 36) {
      RCLCPP_ERROR(get_logger(),
        "Invalid init.P size: %ld (expected 36)", init_P_vec.size());
      return false;
    }
    ekf_.setCovariance(kalman::Covariance<State>(init_P_vec.data()));

    RCLCPP_INFO(get_logger(), "Parameters and EKF initialized successfully");
    RCLCPP_INFO(get_logger(), "Input topics: %s, %s, %s",
      imu_input_topic_.c_str(), gps_input_topic_.c_str(), vel_input_topic_.c_str());
    RCLCPP_INFO(get_logger(),
      "Predict frequency: %.1f Hz, publish frequency: %.1f Hz, queue size: %d",
      predict_freq_, publish_freq_, queue_size_);

    return true;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during parameter/EKF initialization: %s", e.what());
    return false;
  }
}

void EkfLocalizer::initialize_ros_components()
{
  rclcpp::QoS qos(queue_size_);

  // Create separate callback groups for parallel execution:
  // sensor update callbacks vs. the predict/publish timers.
  sub_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  timer_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = sub_callback_group_;

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_input_topic_, qos,
    std::bind(&EkfLocalizer::imu_callback, this, std::placeholders::_1),
    sub_options);

  gps_sub_ = this->create_subscription<av_msgs::msg::GeoPlanePoint>(
    gps_input_topic_, qos,
    std::bind(&EkfLocalizer::gps_callback, this, std::placeholders::_1),
    sub_options);

  vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    vel_input_topic_, qos,
    std::bind(&EkfLocalizer::vel_callback, this, std::placeholders::_1),
    sub_options);

  predict_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1.0 / predict_freq_ * 1000)),
    std::bind(&EkfLocalizer::predict_callback, this),
    timer_callback_group_);

  publish_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1.0 / publish_freq_ * 1000)),
    std::bind(&EkfLocalizer::publish_callback, this),
    timer_callback_group_);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  RCLCPP_INFO(get_logger(), "ROS components initialized with separate callback groups");
}

void EkfLocalizer::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);

  double yaw;
  tf2::getEulerYPR(msg->orientation, yaw, pitch_, roll_);

  ImuMeasurement z;
  z.theta() = ekf_.limitMeasurementYaw(yaw);
  z.omega() = msg->angular_velocity.z;
  z.alpha() = msg->linear_acceleration.x;

  if (ekf_.update(imu_model_, z)) {
    ekf_.wrapStateYaw();
  } else {
    RCLCPP_INFO(
      get_logger(), "Measurement IMU is over the threshold. Discard this measurement.");
  }
}

void EkfLocalizer::gps_callback(const av_msgs::msg::GeoPlanePoint::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);

  GpsMeasurement z;
  z.x() = msg->local_coordinate.x;
  z.y() = msg->local_coordinate.y;
  alt_ = msg->local_coordinate.z;

  // Use the covariance that GPS provided.
  kalman::Covariance<GpsMeasurement> R = kalman::Covariance<GpsMeasurement>::Zero();
  R.diagonal() << msg->position_covariance.at(0), msg->position_covariance.at(4);
  gps_model_.setCovariance(R);

  if (ekf_.update(gps_model_, z)) {
    ekf_.wrapStateYaw();
  } else {
    RCLCPP_INFO(
      get_logger(), "Measurement GPS is over the threshold. Discard this measurement.");
  }
}

void EkfLocalizer::vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);

  VelMeasurement z;
  z.nu() = msg->twist.linear.x;

  if (ekf_.update(vel_model_, z)) {
    ekf_.wrapStateYaw();
  } else {
    RCLCPP_INFO(
      get_logger(), "Measurement Velocity is over the threshold. Discard this measurement.");
  }
}

void EkfLocalizer::predict_callback()
{
  std::lock_guard<std::mutex> lock(mtx_);

  ekf_.predict(sys_);
  ekf_.wrapStateYaw();
}

void EkfLocalizer::publish_callback()
{
  State s;
  double alt;
  double pitch;
  double roll;

  {
    std::lock_guard<std::mutex> lock(mtx_);
    s = ekf_.getState();
    alt = alt_;
    pitch = pitch_;
    roll = roll_;
  }

  tf2::Vector3 t_current(s.x(), s.y(), alt);
  tf2::Quaternion q_current;
  q_current.setRPY(roll, pitch, s.theta());
  q_current.normalize();

  tf2::Transform map_base_link_trans(q_current, t_current);

  geometry_msgs::msg::TransformStamped map_base_link_tf;
  map_base_link_tf.header.stamp = rclcpp::Node::now();
  map_base_link_tf.header.frame_id = "map";
  map_base_link_tf.child_frame_id = "base_link";
  map_base_link_tf.transform = tf2::toMsg(map_base_link_trans);

  tf_broadcaster_->sendTransform(map_base_link_tf);
}

}  // namespace ekf_localizer
