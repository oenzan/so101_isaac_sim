from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    hardware_mode_arg = DeclareLaunchArgument(
        'hardware_mode',
        default_value='fake',
        description='Use fake hardware'
    )

    moveit_config = MoveItConfigsBuilder("so_100_arm", package_name="so_100_arm").to_moveit_configs()
    # Add the hardware_mode parameter
    moveit_config.robot_description = {
        "hardware_mode": LaunchConfiguration('hardware_mode')
    }

    return LaunchDescription([
        hardware_mode_arg,
        generate_demo_launch(moveit_config)
    ])
