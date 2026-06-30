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

  void EstimateLidarPose(LidarFrameList& lidarFrameList,
						             const Eigen::Matrix4d& exTlb,
						             const Eigen::Vector3d& gravity,
						             nav_msgs::Odometry& debugInfo);

  void Estimate(LidarFrameList& lidarFrameList,
				        const Eigen::Matrix4d& exTlb,
				        const Eigen::Vector3d& gravity);

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
  void thread_map_increment_update();
  void convert_vector_to_double(const LidarFrameList& lidar_frame_list);
  void convert_double_to_vector(LidarFrameList& lidar_frame_list);
	void update_local_map_increment(const PointCloudTypePtr& laserCloudCornerStack,
						   const PointCloudTypePtr& laserCloudSurfStack,
						   const PointCloudTypePtr& laserCloudNonFeatureStack,
						   const Eigen::Matrix4d& transformTobeMapped);

  MapManager* map_manager_ptr_;
  ImuAligner* imu_aligner_ptr_;
  LidarFeatureMatcher* lidar_feature_matcher_ptr_;

  std::mutex map_mutex_;         // map operation mutex
  bool miu_thread_running_flag_; // map increment update thread running flag 
  std::thread miu_thread_;       // map increment update thread
  
  PointCloudTypePtr pcf_corner_for_map_ptr_;
  PointCloudTypePtr pcf_surface_for_map_ptr_;
  PointCloudTypePtr pcf_none_for_map_ptr_;
  Eigen::Matrix4d transform_for_map_;
  unsigned int map_update_id_ = 0;

	double param_p_r_[kSlideWindowsSize][6];
	double param_v_bias_[kSlideWindowsSize][9];

	MarginalizationInfo *last_marginalization_info = nullptr;
	std::vector<double *> last_marginalization_parameter_blocks;
	std::vector<PointCloudTypePtr> laserCloudCornerLast;
	std::vector<PointCloudTypePtr> laserCloudSurfLast;
	std::vector<PointCloudTypePtr> laserCloudNonFeatureLast;

	std::vector<PointCloudTypePtr> laserCloudCornerStack;
	std::vector<PointCloudTypePtr> laserCloudSurfStack;
	std::vector<PointCloudTypePtr> laserCloudNonFeatureStack;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeNonFeatureFromLocal;

  PointCloudTypePtr pcf_corner_from_local_;
	PointCloudTypePtr pcf_surface_from_local_;
	PointCloudTypePtr pcf_none_from_local_;
  
	pcl::VoxelGrid<PointType> down_size_filter_corner_;
	pcl::VoxelGrid<PointType> down_size_filter_surface_;
	pcl::VoxelGrid<PointType> down_size_filter_none_;

	pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
	pcl::KdTreeFLANN<PointType> SurfKdMap[10000];
	pcl::KdTreeFLANN<PointType> NonFeatureKdMap[10000];

	pcl::PointCloud<PointType> GlobalSurfMap[10000];
	pcl::PointCloud<PointType> GlobalCornerMap[10000];
	pcl::PointCloud<PointType> GlobalNonFeatureMap[10000];

	int laserCenWidth_last = 10;
	int laserCenHeight_last = 5;
	int laserCenDepth_last = 10;

	static constexpr size_t kLocalMapWindowSize = 50;
	size_t local_map_id_ = 0;
	PointCloudTypePtr local_corner_map_[kLocalMapWindowSize];
	PointCloudTypePtr local_surface_map_[kLocalMapWindowSize];
	PointCloudTypePtr local_none_map_[kLocalMapWindowSize];

	static constexpr unsigned int kMapSkipFrameCount = 2;
	double plan_weight_tan = 0.0;
	double thres_dist = 1.0;
};

} // end of namespace lio

#endif // LIO_POSE_ESTIMATOR_HPP
