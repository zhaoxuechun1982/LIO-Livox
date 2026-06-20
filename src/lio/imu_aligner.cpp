#include "lio/imu_aligner.h"
#include "utils/ceres_utils.hpp"
#include "sophus/so3.hpp"
#include <ros/ros.h>


namespace lio {

ImuAligner* ImuAligner::instance_ptr_ = nullptr;

ImuAligner* ImuAligner::instance_pointer(void)
{
  if (instance_ptr_ == nullptr) {
    instance_ptr_ = new ImuAligner();
  }
  
  return instance_ptr_;
}

ImuAligner& ImuAligner::instance(void)
{
  return *instance_pointer();
}

void ImuAligner::destroy_instance(void)
{
  if (instance_ptr_ != nullptr) {
    delete instance_ptr_;
    instance_ptr_ = nullptr;
  }
}

ImuAligner::ImuAligner(): 
  is_finished_(false),
  opt_result_()
{
}

const Eigen::Vector3d ImuAligner::optimized_ra(void) const
{
  check_cache_valid();
  return opt_result_.ra;
}

const Eigen::Vector3d ImuAligner::optimized_ba(void) const
{
  check_cache_valid();
  return opt_result_.ba;
}

const Eigen::Vector3d ImuAligner::optimized_bg(void) const
{
  check_cache_valid();
  return opt_result_.bg;
}

const std::vector<Eigen::Vector3d> ImuAligner::optimized_v_list(void) const
{
  check_cache_valid();
  return opt_result_.v_list;
}

bool ImuAligner::run_align(std::deque<LidarFrame>& frames,
                           Eigen::Vector3d& g_b,
                           const Eigen::Matrix3d& ex_r_lb, 
                           const Eigen::Vector3d& ex_t_bl,
                           const int& win_size)
{
  // Prevent multiple alignment execution during runtime
  if (is_finished_) {
    ROS_WARN("[ImuAligner][run_align] IMU initial alignment can only run once!");
    return false;
  }

  if (frames.size() < 2) {
    ROS_WARN("[ImuAligner][run_align] Input frames size is not enough!");
    return false;
  }
  
  // Coarse gravity alignment to get rough body-world rotation
  Eigen::Quaterniond q_wg = solve_gravity_rotation(frames);

  // Generate rotation manifold vector
  Eigen::Vector3d prior_r = Sophus::SO3d(q_wg.toRotationMatrix()).log();

  // Generate prior velocity list
  std::vector<Eigen::Vector3d> prior_v_list = calculate_prior_velocity(frames, ex_t_bl);

  // Initialize optimization result container (zero initialized via constructor)
  if (!solve_ceres_problem(frames, q_wg, prior_r, prior_v_list)) {
    return false;
  }

  // Check solved states against physical threshold constraints
  if (!check_optimization_valid(prior_v_list)) {
    return false;
  }

  // Write optimized bias and velocity data back to frame buffer
  if(!write_optimization(frames, g_b)) {
    return false;
  }

  // Recompute all IMU pre-integration segments with updated bias
  refresh_preintegration(frames, ex_r_lb, ex_t_bl, win_size);

  // Mark alignment flow finished to block repeated calls
  is_finished_ = true;

  return true;
}

void ImuAligner::check_cache_valid(void) const
{
  if (!is_finished_) {
    ROS_WARN("[ImuAligner][check_cache_valid] Alignment has not finished, cache data is invalid!");
  }
}

Eigen::Quaterniond ImuAligner::solve_gravity_rotation(const std::deque<LidarFrame>& frames) const
{  
  Eigen::Vector3d avg_acc = -frames.begin()->imuIntegrator.GetAverageAcc();
  avg_acc *= kGravity / avg_acc.norm();

  double q_wg[4] = {1.0, 0.0, 0.0, 0.0};
  ceres::LocalParameterization *manifold_constraint_ptr = new ceres::QuaternionParameterization();
  ceres::Problem problem_q;
  problem_q.AddParameterBlock(q_wg, 4, manifold_constraint_ptr);
  problem_q.AddResidualBlock(Cost_Initial_G::Create(avg_acc), nullptr, q_wg);

  ceres::Solver::Options options_q;
  ceres::Solver::Summary summary_q;
  ceres::Solve(options_q, &problem_q, &summary_q);

  if (!summary_q.IsSolutionUsable())
  {
    ROS_WARN("[ImuAligner][solve_gravity_rotation] Gravity solve diverged, fallback to identity quaternion!");
    return Eigen::Quaterniond::Identity();
  }

  return Eigen::Quaterniond(q_wg[0], q_wg[1], q_wg[2], q_wg[3]);
}

std::vector<Eigen::Vector3d> ImuAligner::calculate_prior_velocity(const std::deque<LidarFrame>& frames, 
                                                                  const Eigen::Vector3d& ex_t_bl) const
{
  std::vector<Eigen::Vector3d> v_list(frames.size());
  if (frames.size() < 2) {
    return v_list;
  }
  
  for (size_t i = 1; i < frames.size(); i++) {
    double dt = frames[i].timeStamp - frames[i-1].timeStamp;
    if (dt > 1e-9) { 
      // Transform IMU body origin to lidar origin in world frame, 
      // compute displacement difference to calculate lidar linear velocity
      v_list[i] = (frames[i].P - frames[i-1].P + frames[i].Q * ex_t_bl - frames[i-1].Q * ex_t_bl) / dt;
    }
    else {
      ROS_ERROR("[ImuAligner][calculate_prior_velocity] Frame timestamp disorder or zero time interval, reuse last valid velocity");
      v_list[i] = v_list[i-1];
    }
  }

  v_list[0] = v_list[1];

  return v_list;
}

bool ImuAligner::solve_ceres_problem(const std::deque<LidarFrame>& frames,
                                     const Eigen::Quaterniond& q_wg,
                                     const Eigen::Vector3d& prior_r,
                                     const std::vector<Eigen::Vector3d>& prior_v_list)
{
  int v_list_size = static_cast<int>(frames.size());
  if (v_list_size < 2) {
    ROS_WARN("[ImuAligner][solve_ceres_problem] Input frames size is not enough!");
    return false;
  }
  if (prior_v_list.size() != v_list_size) {
    ROS_WARN("[ImuAligner][solve_ceres_problem] Input frames size is not equal to prior velocity list size!");
    return false;
  }
  
  // Build ceres cost graph and run optimization
  double para_r[3] = {0.0, 0.0, 0.0};
  double para_ba[3] = {0.0, 0.0, 0.0};
  double para_bg[3] = {0.0, 0.0, 0.0};
  // double para_v_list[v_list_size][3] = {0.0};
  double (*para_v_list)[3] = new double[v_list_size][3]{};

  for (int i = 0; i < v_list_size; i++) {
    for (int j = 0; j < 3; j++) {
      para_v_list[i][j] = prior_v_list[i][j];
    }
  }

  Eigen::Matrix<double, 3, 3> sqrt_information_r  = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_v  = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(para_r, 3);
  problem.AddParameterBlock(para_ba, 3);
  problem.AddParameterBlock(para_bg, 3);

  for(int i = 0; i < v_list_size; i++) {
    problem.AddParameterBlock(para_v_list[i], 3);
  }
  
  // Add CostFunction
  problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r), nullptr, para_r);
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba), nullptr, para_ba);
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg), nullptr, para_bg);
  for(int i = 0; i < v_list_size; i++) {
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v_list[i], sqrt_information_v), nullptr, para_v_list[i]);
  }

  for(int i = 1; i < v_list_size; i++) {
    auto iter = frames.begin();
    auto iter_next = frames.begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    //Eigen::Vector3d pi = iter->P + iter->Q*exPlb;
    //Sophus::SO3d SO3_Ri(iter->Q*exRlb);
    Eigen::Vector3d pi = iter->P + iter->Q * ex_t_bl;
    Sophus::SO3d SO3_Ri(iter->Q * ex_r_lb);
    Eigen::Vector3d ri = SO3_Ri.log();
    //Eigen::Vector3d pj = iter_next->P + iter_next->Q*exPlb;
    //Sophus::SO3d SO3_Rj(iter_next->Q*exRlb);
    Eigen::Vector3d pj = iter_next->P + iter_next->Q * ex_t_bl;
    Sophus::SO3d SO3_Rj(iter_next->Q * ex_r_lb);
    Eigen::Vector3d rj = SO3_Rj.log();

    problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                                                   ri,
                                                                   rj,
                                                                   pj-pi,
                                                                   Eigen::LLT<Eigen::Matrix<double, 9, 9>>
                                                                    (iter_next->imuIntegrator.GetCovariance().block<9,9>(0,0).inverse())
                                                                    .matrixL().transpose()),
                             nullptr,
                             para_r,
                             para_v_list[i-1],
                             para_v_list[i],
                             para_ba,
                             para_bg);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 6;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (!summary.IsSolutionUsable()) {
    ROS_WARN("[ImuAligner][solve_ceres_problem] Optimized result solve diverged!");
    delete[] para_v_list;
    return false;
  }

  opt_result_.ra = Sophus::SO3d::exp(Eigen::Vector3d(para_r[0], para_r[1], para_r[2])) * Eigen::Vector3d(0, 0, -kGravity);
  opt_result_.ba = Eigen::Vector3d(para_ba[0], para_ba[1], para_ba[2]);
  opt_result_.bg = Eigen::Vector3d(para_bg[0], para_bg[1], para_bg[2]);
  for (int i = 0; i < v_list_size; i++) {
    opt_result_.v_list[i] = Eigen::Vector3d(para_v_list[i][0], para_v_list[i][1], para_v_list[i][2]);
  }
  
  delete[] para_v_list;
  return true;
}

bool ImuAligner::check_optimization_valid(const std::vector<Eigen::Vector3d>& prior_v_list)
{
  // Validate bias, velocity and rotation within physical reasonable range
  if (opt_result_.ba.norm() > kBANormThreshold) {
    ROS_WARN("[ImuAligner][check_optimization_valid] Too large accelerometer biases!");
    opt_result_.reset();
    return false;
  }

  if (opt_result_.bg.norm() > kBGNormThreshold) {
    ROS_WARN("[ImuAligner][check_optimization_valid] Too large gyroscope biases!");
    opt_result_.reset();
    return false;
  }
  
  if (prior_v_list.size() != opt_result_.v_list.size()) {
    ROS_WARN("[ImuAligner][check_optimization_valid] Optimized velocity list length is not equal to the prior!");
    opt_result_.reset();
    return false;
  }

  if (opt_result_.v_list.empty()) {
    ROS_WARN("[ImuAligner][check_optimization_valid] Optimized and prior velocity list length is empty!");
    opt_result_.reset();
    return false;
  }
  
  for(int i = 0; i < opt_result_.v_list.size(); i++) {
    if ((opt_result_.v_list[i] - prior_v_list[i]).norm() > kDeltaVelocityNormThreshold) {
      ROS_WARN("[ImuAligner][check_optimization_valid] Too large delta velocity norm!");
      opt_result_.reset();
      return false;
    }
  }

  return true;
}

bool ImuAligner::write_optimization(std::deque<LidarFrame>& frames, Eigen::Vector3d& g_b)
{
  if (frames.empty()) {
    ROS_WARN("[ImuAligner][write_optimization_to_frames] Input frames is empty!");
    opt_result_.reset();
    return false;
  }

  if (frames.size() != opt_result_.v_list.size()) {
    ROS_ERROR("[ImuAligner][write_optimization_to_frames] Input frames size not match optimized velocity list and skip write!");
    opt_result_.reset();
    return false;
  }

  for(size_t i = 0; i < opt_result_.v_list.size(); i++) {
    frames[i].ba = opt_result_.ba;
    frames[i].bg = opt_result_.bg;
    frames[i].V = opt_result_.v_list[i];
  }

  g_b = Sophus::SO3d(opt_result_.ra) * Eigen::Vector3d(0, 0, -kGravity);

  return true;
}

void ImuAligner::refresh_preintegration(std::deque<LidarFrame>& frames,
                                        const Eigen::Matrix3d& ex_r_lb,
                                        const Eigen::Vector3d& ex_t_bl,
                                        const int& win_size) const
{
  if (frames.empty())
  {
    ROS_ERROR("[ImuAligner][refresh_preintegration] Input frames is empty!");
    return;
  }
  
  for(size_t i = 0; i < frames.size() - 1; i++) {
    frames[i+1].imuIntegrator.PreIntegration(frames[i].timeStamp, frames[i].bg, frames[i].ba);
  }

  while(frames.size() > win_size) {
	  frames.pop_front();
  }

  // Update the last frame's P/Q coordinate base as transition point
  // Before MAP init: all frames store lidar-world pose
  // After MAP init: all subsequent frames switch to IMU body-world pose
  // TODO: Add enum field in LidarFrame to explicitly mark coordinate base, 
  // eliminate implicit conversion risk
	Eigen::Vector3d t_lw = frames.back().P;
	Eigen::Quaterniond q_wl = frames.back().Q;
	frames.back().P = q_wl * ex_t_bl + t_lw; // t_bw = q_wl * (origin_b + ex_t_bl) + t_lw 
	frames.back().Q = q_wl * ex_r_lb;        // q_wb

	// std::cout << "\n=============================\n| Initialization Successful |"<<"\n=============================\n" << std::endl;
}

} // namespace lio