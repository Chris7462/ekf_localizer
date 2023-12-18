from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    gps_imu_node = Node(
        package="gps_imu_fusion",
        executable="gps_imu_node",
        name="gps_imu_fusion"
    )

    trajectory_server_node=Node(
        package="trajectory_server",
        executable="trajectory_server_node",
        name="trajectory_server_node",
        namespace = "oxts",
        parameters=[{
            "target_frame_name": "map",
            "source_frame_name": "base_link",
            "trajectory_update_rate": 10.0,
            "trajectory_publish_rate": 10.0
        }]
    )

    return LaunchDescription([
        gps_imu_node,
        trajectory_server_node
    ])
