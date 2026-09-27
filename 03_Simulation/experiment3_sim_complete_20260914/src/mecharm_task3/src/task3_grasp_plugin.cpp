#include <memory>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <mutex>
#include <gazebo/physics/ContactManager.hh>
#include <gazebo/physics/Contact.hh>
#include <std_msgs/msg/string.hpp>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace mecharm_task3
{
class Task3GraspPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();
    node_ = gazebo_ros::Node::Get(sdf);
    safety_pub_ = node_->create_publisher<std_msgs::msg::String>("/task3/physics_safety", 10);
    world_->Physics()->GetContactManager()->SetNeverDropContacts(true);

    base_link_name_ = sdf->HasElement("base_link") ?
      sdf->Get<std::string>("base_link") : "base";
    gripper_link_name_ = sdf->HasElement("gripper_link") ?
      sdf->Get<std::string>("gripper_link") : "gripper_base";
    target_model_name_ = sdf->HasElement("target_model") ?
      sdf->Get<std::string>("target_model") : "task3_object_1";
    target_link_name_ = sdf->HasElement("target_link") ?
      sdf->Get<std::string>("target_link") : "link";
    grasp_offset_ = sdf->HasElement("grasp_offset") ?
      sdf->Get<ignition::math::Vector3d>("grasp_offset") :
      ignition::math::Vector3d(0.0, -0.10, -0.012);
    max_grasp_error_ = sdf->HasElement("max_grasp_error") ?
      sdf->Get<double>("max_grasp_error") : 0.025;
    close_threshold_ = sdf->HasElement("close_threshold") ?
      sdf->Get<double>("close_threshold") : -0.20;
    close_when_greater_ = sdf->HasElement("close_when_greater") ?
      sdf->Get<bool>("close_when_greater") : false;
    minimum_target_height_ = sdf->HasElement("minimum_target_height") ?
      sdf->Get<double>("minimum_target_height") : 0.018;

    node_->declare_parameter<std::string>("target_model", target_model_name_);
    selection_callback_ = node_->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        rcl_interfaces::msg::SetParametersResult result; result.successful = true;
        for (const auto & parameter : parameters) {
          if (parameter.get_name() != "target_model") {continue;}
          if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING || grasp_joint_) {
            result.successful = false; result.reason = "Invalid target type or an object is still held"; return result;
          }
          const auto name = parameter.as_string();
          if (name.find("task3_object_") != 0 || !world_->ModelByName(name)) {
            result.successful = false; result.reason = "Requested object does not exist"; return result;
          }
          std::lock_guard<std::mutex> lock(selection_mutex_); target_model_name_ = name;
        }
        return result;
      });

    primary_gripper_joint_ = model_->GetJoint("gripper_controller");
    const std::vector<std::pair<std::string, double>> mimic_specs = {
      {"gripper_base_to_gripper_left2", 1.0},
      {"gripper_left3_to_gripper_left1", -1.0},
      {"gripper_base_to_gripper_right3", -1.0},
      {"gripper_base_to_gripper_right2", -1.0},
      {"gripper_right3_to_gripper_right1", 1.0},
    };
    for (const auto & spec : mimic_specs) {
      const auto joint = model_->GetJoint(spec.first);
      if (joint) {
        mimic_joints_.emplace_back(joint, spec.second);
      } else {
        RCLCPP_ERROR(node_->get_logger(), "Mimic joint '%s' was not found", spec.first.c_str());
      }
    }
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&Task3GraspPlugin::EnforceMimicJoints, this));

    const auto base_link = model_->GetLink(base_link_name_);
    if (base_link) {
      anchor_joint_ = world_->Physics()->CreateJoint("fixed", model_);
      anchor_joint_->SetName("task2_world_anchor");
      anchor_joint_->Load(gazebo::physics::LinkPtr(), base_link, ignition::math::Pose3d());
      anchor_joint_->Init();
      RCLCPP_INFO(node_->get_logger(), "mechArm base anchored to the Gazebo world");
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Base link '%s' was not found", base_link_name_.c_str());
    }

    attach_service_ = node_->create_service<std_srvs::srv::Trigger>(
      "task3/attach",
      std::bind(&Task3GraspPlugin::Attach, this, std::placeholders::_1, std::placeholders::_2));
    detach_service_ = node_->create_service<std_srvs::srv::Trigger>(
      "task3/detach",
      std::bind(&Task3GraspPlugin::Detach, this, std::placeholders::_1, std::placeholders::_2));
    check_service_ = node_->create_service<std_srvs::srv::Trigger>(
      "task3/check_grasp",
      std::bind(&Task3GraspPlugin::CheckGrasp, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  void EnforceMimicJoints()
  {
    std::string selected;
    {std::lock_guard<std::mutex> lock(selection_mutex_); selected = target_model_name_;}
    const auto manager = world_->Physics()->GetContactManager();
    const auto & contacts = manager->GetContacts();
    for (unsigned int i = 0; i < manager->GetContactCount(); ++i) {
      const auto c = contacts[i];
      const auto a = c->collision1->GetLink();
      const auto b = c->collision2->GetLink();
      const bool arm_a = a->GetModel() == model_;
      const bool arm_b = b->GetModel() == model_;
      if (arm_a == arm_b) {continue;}
      const auto arm = arm_a ? a : b;
      const auto other = arm_a ? b : a;
      if (arm->GetName() == "base" && other->GetModel()->GetName() == "task2_table") {continue;}
      if (arm->GetName().find("gripper") != std::string::npos && other->GetModel()->GetName() == selected) {continue;}
      for (int k = 0; k < c->count; ++k) {
        if (c->depths[k] > 0.002) {
          unsafe_pairs_.insert(arm->GetName() + "|" + other->GetModel()->GetName());
        }
      }
    }
    const double now = world_->SimTime().Double();
    if (now - last_safety_publish_ > 0.5) {
      std_msgs::msg::String msg;
      std::ostringstream json;
      json << "{\"unsafe_contact_pairs\":" << unsafe_pairs_.size() << ",\"pairs\":[";
      bool first = true;
      for (const auto & pair : unsafe_pairs_) {
        if (!first) {json << ",";} first = false; json << "\"" << pair << "\"";
      }
      json << "]}"; msg.data = json.str(); safety_pub_->publish(msg); last_safety_publish_ = now;
    }
    if (!primary_gripper_joint_) {
      return;
    }
    const double primary = primary_gripper_joint_->Position(0);
    for (const auto & item : mimic_joints_) {
      item.first->SetPosition(0, item.second * primary, true);
    }
  }

  bool EvaluateGrasp(std::string & status)
  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    const auto gripper_link = model_->GetLink(gripper_link_name_);
    const auto gripper_joint = model_->GetJoint("gripper_controller");
    const auto target_model = world_->ModelByName(target_model_name_);
    const auto target_link = target_model ? target_model->GetLink(target_link_name_) : nullptr;
    if (!gripper_link || !gripper_joint || !target_link) {
      status = "gripper joint/link or target link was not found";
      return false;
    }

    const auto target_in_gripper =
      gripper_link->WorldPose().Inverse() * target_link->WorldPose();
    const auto error = target_in_gripper.Pos() - grasp_offset_;
    const double distance = error.Length();
    const double gripper_position = gripper_joint->Position(0);
    const double target_height = target_link->WorldPose().Pos().Z();
    const bool centered = distance <= max_grasp_error_;
    const bool closed = close_when_greater_ ?
      gripper_position >= close_threshold_ : gripper_position <= close_threshold_;
    const bool above_table = target_height >= minimum_target_height_;

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4)
           << "center_error=" << distance
           << " m, local_error=[" << error.X() << ", " << error.Y() << ", " << error.Z() << "]"
           << ", gripper=" << gripper_position
           << " rad, target_z=" << target_height
           << " m, centered=" << (centered ? "true" : "false")
           << ", closed=" << (closed ? "true" : "false")
           << ", above_table=" << (above_table ? "true" : "false");
    status = stream.str();
    return centered && closed && above_table;
  }

  void CheckGrasp(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    response->success = EvaluateGrasp(response->message);
  }

  void Attach(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (grasp_joint_) {
      response->success = true;
      response->message = "target is already attached";
      return;
    }

    std::string evaluation;
    if (!EvaluateGrasp(evaluation)) {
      response->success = false;
      response->message = "grasp rejected: " + evaluation;
      RCLCPP_WARN(node_->get_logger(), "%s", response->message.c_str());
      return;
    }

    const auto gripper_link = model_->GetLink(gripper_link_name_);
    const auto target_model = world_->ModelByName(target_model_name_);
    const auto target_link = target_model ? target_model->GetLink(target_link_name_) : nullptr;
    if (!gripper_link || !target_link) {
      response->success = false;
      response->message = "gripper or target link was not found";
      return;
    }

    grasp_joint_ = world_->Physics()->CreateJoint("fixed", model_);
    grasp_joint_->SetName("task2_grasp_joint");
    grasp_joint_->Load(gripper_link, target_link, ignition::math::Pose3d());
    grasp_joint_->Init();
    response->success = true;
    response->message = "target attached without pose snapping; " + evaluation;
    RCLCPP_INFO(node_->get_logger(), "%s", response->message.c_str());
  }

  void Detach(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!grasp_joint_) {
      response->success = true;
      response->message = "target is already detached";
      return;
    }
    grasp_joint_->Detach();
    grasp_joint_.reset();
    response->success = true;
    response->message = "target detached from adaptive gripper";
    RCLCPP_INFO(node_->get_logger(), "%s", response->message.c_str());
  }

  gazebo_ros::Node::SharedPtr node_;
  std::mutex selection_mutex_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr selection_callback_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr safety_pub_;
  std::set<std::string> unsafe_pairs_;
  double last_safety_publish_ = 0;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::ModelPtr model_;
  gazebo::physics::JointPtr anchor_joint_;
  gazebo::physics::JointPtr grasp_joint_;
  gazebo::physics::JointPtr primary_gripper_joint_;
  std::vector<std::pair<gazebo::physics::JointPtr, double>> mimic_joints_;
  gazebo::event::ConnectionPtr update_connection_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr attach_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr detach_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr check_service_;
  std::string base_link_name_;
  std::string gripper_link_name_;
  std::string target_model_name_;
  std::string target_link_name_;
  ignition::math::Vector3d grasp_offset_;
  double max_grasp_error_;
  double close_threshold_;
  bool close_when_greater_;
  double minimum_target_height_;
};
}  // namespace mecharm_task3

GZ_REGISTER_MODEL_PLUGIN(mecharm_task3::Task3GraspPlugin)
