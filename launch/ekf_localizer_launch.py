from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',  # or 'true' if appropriate
        description='Use simulation time'
    )

    params = join(
        get_package_share_directory('ekf_localizer'), 'param',
        'ekf_localizer.yaml'
    )

    ekf_localizer_node = Node(
        package='ekf_localizer',
        executable='ekf_localizer_node',
        name='ekf_localizer_node',
        parameters=[
            params,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        ekf_localizer_node
    ])
