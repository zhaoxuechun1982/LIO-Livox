#ifndef LIO_POSE_ESTIMATOR_HPP
#define LIO_POSE_ESTIMATOR_HPP

#include "type/point_cloud_type.hpp"
#include "type/lidar_frame_type.hpp"
#include "type/lidar_feature_type.hpp"
#include "utils/ceres_utils.hpp"
#include "lio/imu_aligner.hpp"
#include "lio/imu_integrator.hpp"
#include "lio/lidar_feature_matcher.hpp"
#include "lio/map_manager.hpp"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>

#include <queue>
#include <iterator>
#include <future>
#include <chrono>

using namespace lio_data_type;
namespace lio
{
class PoseEstimator 
{
public:
	static constexpr unsigned int kSlideWindowsSize = 2;

public:
  PoseEstimator(const float& filter_corner, const float& filter_surface);
  ~PoseEstimator();

  void estimate_lidar_pose(LidarFrameList& lidarFrameList,
						   const Eigen::Matrix4d& exTlb,
						   const Eigen::Vector3d& gravity,
						   nav_msgs::Odometry& debugInfo);

  bool run_imu_align(LidarFrameDeque& frames,
                     Eigen::Vector3d& g_b,
                     const Eigen::Matrix3d& ex_r_lb, 
                     const Eigen::Vector3d& ex_t_bl);

  inline PointCloudTypePtr get_corner_map()
  {
	return map_manager_ptr_->get_corner_map();
  }
	
  inline PointCloudTypePtr get_surf_map()
  {
	return map_manager_ptr_->get_surf_map();
  }

  inline PointCloudTypePtr get_nonfeature_map()
  {
	return map_manager_ptr_->get_nonfeature_map();
  }

private:
  friend class LidarFeatureMatcher;
  void estimate(LidarFrameList& lidarFrameList,
				const Eigen::Matrix4d& exTlb,
				const Eigen::Vector3d& gravity);

  void thread_map_increment_update();
  void convert_vector_to_double(const LidarFrameList& lidar_frame_list);
  void convert_double_to_vector(LidarFrameList& lidar_frame_list);
  void update_local_map_increment(const PointCloudTypePtr& laserCloudCornerStack,
						                      const PointCloudTypePtr& laserCloudSurfStack,
						                      const PointCloudTypePtr& laserCloudNonFeatureStack,
						                      const Eigen::Matrix4d& transformTobeMapped);

  // To do...to reserve										  
  MapManager* map_manager_ptr_;
  ImuAligner* imu_aligner_ptr_;
  LidarFeatureMatcher lidar_feature_matcher_;

  double param_p_r_[kSlideWindowsSize][6];
  double param_v_bias_[kSlideWindowsSize][9];

  MarginalizationInfo *last_marginalization_info = nullptr;
  std::vector<double *> last_marginalization_parameter_blocks;

  static constexpr unsigned int kMapSkipFrameCount = 2;
  double plan_weight_tan = 0.0;
  double thres_dist = 1.0;

  static constexpr size_t kValidVoxelGridCount = 4851;

  // to move
  std::mutex map_mutex_;         // map operation mutex
  bool miu_thread_running_flag_; // map increment update thread running flag 
  std::thread miu_thread_;       // map increment update thread
  unsigned int map_update_id_ = 0;
  Eigen::Matrix4d transform_for_map_;
  
  PointCloudTypePtr pcf_corner_for_map_ptr_;
  PointCloudTypePtr pcf_surface_for_map_ptr_;
  PointCloudTypePtr pcf_none_for_map_ptr_;

  std::vector<PointCloudTypePtr> laserCloudCornerLast;
  std::vector<PointCloudTypePtr> laserCloudSurfLast;
  std::vector<PointCloudTypePtr> laserCloudNonFeatureLast;

  std::vector<PointCloudTypePtr> pcf_corner_stack_;
  std::vector<PointCloudTypePtr> pcf_surface_stack_;
  std::vector<PointCloudTypePtr> pcf_none_stack_;

  PointKdTreeTypePtr kdtree_corner_from_local_ptr_;
  PointKdTreeTypePtr kdtree_surface_from_local_ptr_;
  PointKdTreeTypePtr kdtree_none_from_local_ptr_;

  PointCloudTypePtr pcf_corner_from_local_;
  PointCloudTypePtr pcf_surface_from_local_;
  PointCloudTypePtr pcf_none_from_local_;

  static constexpr size_t kLocalMapWindowSize = 50;
  size_t local_map_id_ = 0;
  PointCloudTypePtr local_corner_map_[kLocalMapWindowSize];
  PointCloudTypePtr local_surface_map_[kLocalMapWindowSize];
  PointCloudTypePtr local_none_map_[kLocalMapWindowSize]; 

  // Down sampleing filter 
  PointVoxelGridType down_size_corner_filter_;
  PointVoxelGridType down_size_surface_filter_;
  PointVoxelGridType down_size_none_filter_;

  // To do...to remove
  static constexpr size_t kGlobalMapWindowSize = 10000;
  PointKdTreeType kdtree_corner_map_[kGlobalMapWindowSize];
  PointKdTreeType kdtree_surface_map_[kGlobalMapWindowSize];
  PointKdTreeType kdtree_none_map_[kGlobalMapWindowSize];
  PointCloudType global_corner_map_[kGlobalMapWindowSize];
  PointCloudType global_surface_map_[kGlobalMapWindowSize];
  PointCloudType global_none_map_[kGlobalMapWindowSize];
  int laser_center_width_last_ = 10;
  int laser_center_height_last_ = 5;
  int laser_center_depth_last_ = 10;
  
};

} // end of namespace lio

#endif // LIO_POSE_ESTIMATOR_HPP
