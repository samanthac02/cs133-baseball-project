'''
display.launch.py

Launch the G1 robot description and RVIZ.
Run the trajectory separately with:

    ros2 run g1_baseball trajectory
'''

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Find the package.
    packagepath = Path(
        get_package_share_directory('g1_baseball')
    )

    # Load the URDF.
    urdf_file = packagepath / 'urdf' / 'g1_baseball.urdf'
    robot_description = urdf_file.read_text()

    return LaunchDescription([
        # Convert joint states into robot link transforms.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': 100.0,
            }],
            output='screen'
        ),

        # Open RVIZ.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])