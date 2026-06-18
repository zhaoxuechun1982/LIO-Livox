#ifndef POINTS_CORRECT_HPP
#define POINTS_CORRECT_HPP

namespace lidar_proc
{
/**
 * @brief Static service module for ground plane estimation and lidar point correction
 * @details Designed with global unique internal state, consistent with original project logic.
 *          All persistent ground plane data are stored as static class members.
 */
class PointsCorrect
{
public:
  /**
   * @brief Main entrance for ground plane estimation with temporal smoothing
   * @param[out] pos_out Float array with fixed length 6
   *             pos_out[0], pos_out[1], pos_out[2]: smoothed ground normal vector
   *             pos_out[3], pos_out[4], pos_out[5]: average ground patch center
   * @param[in] points_in Raw input points buffer, each point occupies 4 float [x,y,z,reserved]
   * @param[in] points_num Total number of input points
   * @return Valid ground patch count used for averaging; return 0 if no valid ground patch
   * @warning The external buffer pos_out must allocate at least 6 float elements
   */
  static int estimate_and_smooth_ground(float* pos_out, float* points_in, int points_num);

  /**
   * @brief Reset all global persistent ground state to initial zero value
   * @details Used for scene switch or abnormal frame recovery
   */
  static void reset(void);

private:
  /**
   * @brief Filter candidate ground points via grid height statistical constraint
   * @param[out] points_out Filtered ground candidate float buffer, 4 floats per point
   * @param[in] points_in Raw input point float buffer
   * @param[in] points_num Total input point count
   * @return Number of filtered ground candidate points
   */
  static int filter_ground_candidate(float* points_out, float* points_in, int points_num);

  /**
   * @brief Traverse the filtered ground points, calculate average ground plane pos via local PCA
   * @param[out] ground_pos_out Output average ground plane pos data [nx, ny, nz, cx, cy, cz]
   * @param[in] points_in Ground candidate point float buffer
   * @param[in] points_num Total count of ground candidate points
   * @param[in] search_radius Radius for neighborhood PCA search in meter
   * @return Number of valid flat ground patches involved in average calculation
   */
  static int calculate_ground_pos(float* ground_pos_out, float* points_in, int points_num, float search_radius);

  /**
   * @brief Correct raw point cloud coordinate based on estimated ground plane
   * @param[in][out] points Raw point float buffer that will be modified in-place
   * @param[in] point_num Total input point count
   * @param[in] ground_pos Reference ground plane parameter [nx, ny, nz, cx, cy, cz]
   * @return Fixed return value 0
   */
  static int correct_point_cloud(float* points, int point_num, float* ground_pos);

  // Persistent smoothed ground plane parameters [nx, ny, nz, cx, cy, cz]
  static float m_gound_pos[6];
  
  // Continuous stable frame counter for temporal smoothing logic
  static int m_frame_count;  
  
  // Frame count threshold to decide refresh or smooth ground plane
  static constexpr int FRAME_COUNT_THRESHOLD = 10;
  
  // Eigen value ratio threshold to judge a neighborhood as flat ground plane
  static constexpr float PCA_PLANE_RATIO_THR = 10.0f;
  
  // Max capacity of temporary ground candidate point buffer
  static constexpr int MAX_GND_CANDIDATE_NUM = 60000;
  
  // Fixed neighborhood search radius for ground patch PCA calculation, unit: m
  static constexpr float PCA_SEARCH_RADIUS = 1.0f;
  
  // Normal vector deviation threshold for frame smoothing judgment
  static constexpr float NORMAL_DIFF_THR = 0.1f; 
  
  // Max count of ground patches to avoid excessive computation
  static constexpr int MAX_GROUND_PATCH_LIMIT = 1000;
  // Eigen value ratio threshold to judge flat ground plane
  static constexpr float GROUND_PLANE_EIGEN_RATIO_THR = 5000.0f;
  
  // Ground normal X component correction thresholds
  static constexpr float NORM_X_CORR_THR1 = 0.1f;
  static constexpr float NORM_X_CORR_THR2 = 0.2f;

  // Piecewise scale factor for X correction
  static constexpr float NORM_X_SCALE_LIGHT = 4.5f;
  static constexpr float NORM_X_SCALE_MID = 3.2f;
  static constexpr float NORM_X_SCALE_HEAVY = 2.8f;

  // Fixed scale factor for Y component
  static constexpr float NORM_Y_FIX_SCALE = 2.3f;
};

} // namespace lidar_proc

#endif // POINTS_CORRECT_HPP