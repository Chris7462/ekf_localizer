from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = join(
        get_package_share_directory('ekf_localizer'), 'params', 'ekf_localizer.yaml'
    )

    ekf_localizer_node = Node(
        package='ekf_localizer',
        executable='ekf_localizer_node',
        name='ekf_localizer_node',
        parameters=[params]
    )

    return LaunchDescription([
        ekf_localizer_node
    ])
