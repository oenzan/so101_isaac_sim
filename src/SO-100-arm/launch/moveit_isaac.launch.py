from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    hardware_mode_arg = DeclareLaunchArgument(
        'hardware_mode',
        default_value='isaac',
        description='fake | isaac | real'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use simulation time (/clock)"
    )

    joint_commands_topic_arg = DeclareLaunchArgument(
        'joint_commands_topic',
        default_value='/isaac_joint_commands',
        description='joint commands topic for Isaac Sim'
    )

    joint_states_topic_arg = DeclareLaunchArgument(
        'joint_states_topic',
        default_value='/isaac_joint_states',
        description='joint states topic from Isaac Sim'
    )

    moveit_config = (
        MoveItConfigsBuilder("so_100_arm", package_name="so_100_arm")
        .robot_description(mappings={
            "hardware_mode": LaunchConfiguration("hardware_mode"),
            "joint_commands_topic": LaunchConfiguration("joint_commands_topic"),
            "joint_states_topic": LaunchConfiguration("joint_states_topic"),
        })
        .to_moveit_configs()
    )

    return LaunchDescription([
        hardware_mode_arg,
        use_sim_time_arg,
        joint_commands_topic_arg,
        joint_states_topic_arg,
        generate_demo_launch(moveit_config),
    ])
