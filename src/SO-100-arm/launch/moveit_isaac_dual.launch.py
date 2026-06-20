import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    
    hardware_mode_arg = DeclareLaunchArgument(
        "hardware_mode",
        default_value="isaac",
        description="fake | isaac | real",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time (/clock)",
    )
    hardware_mode = LaunchConfiguration("hardware_mode")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # CONFIG PATH DUZELTILDI: "config/so_100_dual.urdf.xacro"
    moveit_config = (
        MoveItConfigsBuilder("so_100_dual", package_name="so_100_arm")
        .robot_description(
            file_path="config/so_100_dual.urdf.xacro",
            mappings={"hardware_mode": hardware_mode}
        )
        .robot_description_semantic(file_path="config/so_100_dual.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .joint_limits(file_path="config/joint_limits_dual.yaml")
        .to_moveit_configs()
    )

    # NODES
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_config = os.path.join(
        get_package_share_directory("so_100_arm"),
        "config/moveit.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )
    print(moveit_config.to_dict())
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": use_sim_time},
        ],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("so_100_arm"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            ros2_controllers_path,
            {"use_sim_time": use_sim_time}
        ],
        output="both",
    )

    controllers_to_spawn = [
        "joint_state_broadcaster",
        "left_arm_controller",
        "right_arm_controller",
    ]
    load_controllers = []
    for controller in controllers_to_spawn:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(controller)],
                shell=True,
                output="screen",
            )
        ]

    return LaunchDescription(
        [hardware_mode_arg, use_sim_time_arg, rviz_node, robot_state_publisher, move_group_node, ros2_control_node]
        + load_controllers
    )