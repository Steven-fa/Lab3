cmake_minimum_required(VERSION 3.8)
project(mecharm_task2)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)

add_library(task2_grasp_plugin SHARED src/task2_grasp_plugin.cpp)
target_include_directories(task2_grasp_plugin PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(task2_grasp_plugin ${GAZEBO_LIBRARIES})
ament_target_dependencies(task2_grasp_plugin gazebo_ros rclcpp std_srvs)

install(TARGETS task2_grasp_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  config
  launch
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/task2_scene.py
  scripts/probe_task2_ik.py
  scripts/search_grasp_ik.py
  scripts/search_jointspace_grasp.py
  scripts/task2_pick_place.py
  scripts/verify_moveit_execute.py
  scripts/verify_moveit_ik.py
  scripts/verify_moveit_plan.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
