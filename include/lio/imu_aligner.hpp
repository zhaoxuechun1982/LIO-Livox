#ifndef QS_LIO_IMU_ALIGNER_HPP
#define QS_LIO_IMU_ALIGNER_HPP

#include <deque>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lio 
{

/**
 * @brief Global singleton IMU initial aligner based on MAP batch optimization
 * Coarse gravity alignment + fine MAP optimization for IMU bias & initial states
 */
class ImuAligner
{
public:
  /**
   * @brief Get unique global singleton instance
   * @return Reference to static ImuAligner instance pointer or instance
   */
  static ImuAligner& instance(void);
  static ImuAligner* instance_pointer(void);
  static void destroy_instance(void);
  const Eigen::Vector3d& optimized_ra(void) const;
  const Eigen::Vector3d& optimized_ba(void) const;
  const Eigen::Vector3d& optimized_bg(void) const;
  const std::vector<Eigen::Vector3d>& optimized_v_list(void) const;

/**
 * @brief Main entry of full IMU initial alignment pipeline
 * @param[in][out] frames Sliding window buffer of lidar frames with IMU pre-integration
 * @param[out] g_b gravity vector expressed in IMU body frame 
 * @param[in] ex_r_lb Extrinsic rotation matrix, transform point from lidar frame to IMU body frame
 * @param[in] ex_t_bl Extrinsic translation, transform point from IMU body frame to lidar frame
 * @param[in] win_size Sliding lidar frame window size
 * @return True if MAP alignment converges and passes validity check
 */
bool run_align(std::deque<LidarFrame>& frames,
               Eigen::Vector3d& g_b, 
               const Eigen::Matrix3d& ex_r_lb,
               const Eigen::Vector3d& ex_t_bl,
               const int& win_size);

private:
  /**
   * @brief Internal container storing all solved variables from MAP optimization
   */
  struct OptResult
  {
    Eigen::Vector3d ra;                   // Optimized rotation axis-angle vector from world to IMU body frame
    Eigen::Vector3d ba;                   // Optimized accelerometer bias vector
    Eigen::Vector3d bg;                   // Optimized gyroscope bias vector
    std::vector<Eigen::Vector3d> v_list;  // Optimized world-frame velocity of each lidar frame in window

    /**
     * @brief Default constructor, zero initialize all vector states
     */
    OptResult(void) 
    {
      ra.setZero();
      ba.setZero();
      bg.setZero();
    }

    void reset(void)
    {
      ra.setZero(); 
      ba.setZero();
      bg.setZero();
      v_list.clear();      
    }
  };

  /**
   * @brief Private constructor, singleton pattern forbid external instantiation
   */
  ImuAligner();
  ~ImuAligner() = default;
  ImuAligner(const ImuAligner&) = delete;
  ImuAligner& operator=(const ImuAligner&) = delete;

  void check_cache_valid(void) const;

  /**
   * @brief Solve body-to-world rotation using static gravity observation
   * @param[in] frames Sliding lidar frame window
   * @return Rough quaternion from body to world gravity direction
   * @note Coarse alignment
   */
  Eigen::Quaterniond solve_gravity_rotation(const std::deque<LidarFrame>& frames) const;

  /**
   * @brief Solve prior velocity and rotation
   * @param[in] frames Sliding lidar frame window
   * @param[in] ex_t_bl Extrinsic translation, transform point from IMU body frame to lidar frame
   * @return prior velocity list in frames
   */
  std::vector<Eigen::Vector3d> calculate_prior_velocity(const std::deque<LidarFrame>& frames, 
                                                        const Eigen::Vector3d& ex_t_bl) const;

  /**
   * @brief Build Ceres MAP cost function and run nonlinear optimization
   * @param frames[in] Sliding lidar frame window
   * @param q_wg[in] Coarse gravity rotation quaternion
   * @param prior_r[in] Prior rotation manifold value
   * @param prior_v_list[in] Prior velocity list of all frames
   * @return True if optimization converges normally
   */
  bool solve_ceres_problem(const std::deque<LidarFrame>& frames,
                           const Eigen::Quaterniond& q_wg,
                           const Eigen::Vector3d& prior_r,
                           const std::vector<Eigen::Vector3d>& prior_v_list);

  /**
   * @brief Physical threshold validation for optimized IMU states
   * @param [in] prior_v_list Raw velocity prior list for reference
   * @return True if all solved parameters are within reasonable physical bounds
   */
  bool check_optimization_valid(const std::vector<Eigen::Vector3d>& prior_v_list);

  /**
   * @brief Write optimized bias and velocity back to lidar frame buffer
   * @param[out] frames Mutable lidar frame window buffer
   * @param[out] g_b Gravity vector under IMU body frame output
   */
  bool write_optimization(std::deque<LidarFrame>& frames, Eigen::Vector3d& g_b) const;

  /**
   * @brief Re-calculate all IMU pre-integration blocks with updated bias
   * @param[out] frames Sliding lidar frame window
   * @param[in] ex_r_lb Extrinsic rotation matrix, transform point from lidar frame to IMU body frame
   * @param[in] ex_t_bl Extrinsic translation, transform point from IMU body frame to lidar frame
   * @param[in] win_size Sliding lidar frame window size
   */
  void refresh_preintegration(std::deque<LidarFrame>& frames,
                              const Eigen::Matrix3d& ex_r_lb,
                              const Eigen::Vector3d& ex_t_bl,
                              const int& win_size) const;

  bool is_finished_;
  OptResult opt_result_;
  static ImuAligner* instance_ptr_;
  static constexpr double kGravity = 9.805; 
  static constexpr double kBANormThreshold = 0.5;
  static constexpr double kBGNormThreshold = 0.5; 
  static constexpr double kDeltaVelocityNormThreshold = 2.0; 
};

} // namespace lio

#endif // end of QS_LIO_IMU_ALIGNER_HPP
