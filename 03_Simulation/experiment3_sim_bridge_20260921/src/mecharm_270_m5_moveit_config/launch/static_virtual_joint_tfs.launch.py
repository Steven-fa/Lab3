from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_static_virtual_joint_tfs_launch


def generate_launch_description():
    config = MoveItConfigsBuilder("firefighter", package_name="mecharm_270_m5_moveit_config").to_moveit_configs()
    return generate_static_virtual_joint_tfs_launch(config)
