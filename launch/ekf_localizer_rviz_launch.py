from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bag_exec = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-r', '1.0',
             '/data/kitti/raw/2011_09_29_drive_0071_sync_bag', '--clock']
    )

    # The TF and URDF of the vehicle
    robot_state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('kitti_urdf'), 'launch', 'kitti_urdf_launch.py'
            ])
        ])
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', join(get_package_share_directory('ekf_localizer'),
                              'rviz', 'ekf_localizer.rviz')]
    )

    local_gps_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('local_gps_imu'), 'launch', 'local_gps_imu_launch.py'
            ])
        ])
    )

    trajectory_server_gps_node = Node(
        package='trajectory_server',
        executable='trajectory_server_node',
        name='trajectory_server_node',
        namespace='oxts',
        parameters=[{
            'target_frame_name': 'map',
            'source_frame_name': 'oxts_link',
            'trajectory_update_rate': 10.0,
            'trajectory_publish_rate': 10.0
        }]
    )

    ekf_localizer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ekf_localizer'), 'launch', 'ekf_localizer_launch.py'
            ])
        ])
    )

    trajectory_server_ekf_node = Node(
        package='trajectory_server',
        executable='trajectory_server_node',
        name='trajectory_server_node',
        namespace='ekf',
        parameters=[{
            'target_frame_name': 'map',
            'source_frame_name': 'base_link',
            'trajectory_update_rate': 10.0,
            'trajectory_publish_rate': 10.0
        }]
    )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        bag_exec,
        robot_state_publisher_launch,
        rviz_node,
        local_gps_imu_launch,
        ekf_localizer_launch,
        TimerAction(
            period=1.0,  # delay these nodes for 1.0 seconds.
            actions=[
                trajectory_server_gps_node,
                trajectory_server_ekf_node
            ]
        )
    ])
