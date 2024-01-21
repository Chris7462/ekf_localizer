cmake_minimum_required(VERSION 3.8)
project(ekf_localizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(kalman_filter REQUIRED)

# system
find_package(GeographicLib REQUIRED)

add_executable(ekf_localizer_node
  src/ekf_localizer_node.cpp
  src/ekf_localizer.cpp
  src/system_model.cpp
  src/gps_measurement_model.cpp
  src/imu_measurement_model.cpp
  src/vel_measurement_model.cpp
  src/extended_kalman_filter.cpp)

target_include_directories(ekf_localizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(ekf_localizer_node PUBLIC
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
  kalman_filter)

target_compile_features(ekf_localizer_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS ekf_localizer_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY params
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  set(test_sources
    system_model
    gps_measurement_model
    imu_measurement_model
    vel_measurement_model)

  foreach(test_src ${test_sources})
    ament_add_gtest(${test_src}_test
      test/${test_src}_test.cpp
      src/${test_src}.cpp)

    target_include_directories(${test_src}_test PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)

    ament_target_dependencies(${test_src}_test
      kalman_filter)

    target_link_libraries(${test_src}_test gtest_main)
  endforeach()

  ament_add_gtest(extended_kalman_filter_test
    test/extended_kalman_filter_test.cpp
    src/system_model.cpp
    src/gps_measurement_model.cpp
    src/imu_measurement_model.cpp
    src/vel_measurement_model.cpp
    src/extended_kalman_filter.cpp)

  target_include_directories(extended_kalman_filter_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  ament_target_dependencies(extended_kalman_filter_test
    kalman_filter)

  target_link_libraries(extended_kalman_filter_test gtest_main)

endif()

ament_package()
