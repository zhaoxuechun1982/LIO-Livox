#ifndef LIO_POSE_ESTIMATOR_HPP
#define LIO_POSE_ESTIMATOR_HPP

#include "type/lidar_frame.hpp"
#include "type/lidar_feature.hpp"
#include "utils/ceres_utils.hpp"
#include "lio/imu_aligner.hpp"
#include "lio/imu_integrator.hpp"
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

namespace lio
{
class PoseEstimator 
{
public:
	static constexpr unsigned int silde_windows_size = 2;

public:
	PoseEstimator(const float& filter_corner, const float& filter_surface);
	~PoseEstimator();

	[[noreturn]] void threadMapIncrement();
	void processPointToLine(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeatureLine>& vLineFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCornerMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	void processPointToPlan(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeaturePlan>& vPlanFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	void processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
							   std::vector<FeaturePlanVec>& vPlanFeatures,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							   const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							   const Eigen::Matrix4d& exTlb,
							   const Eigen::Matrix4d& m4d);
				
	void processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
							  std::vector<FeatureNon>& vNonFeatures,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
							  const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
							  const Eigen::Matrix4d& exTlb,
							  const Eigen::Matrix4d& m4d);

	void vector2double(const std::list<LidarFrame>& lidarFrameList);
	void double2vector(std::list<LidarFrame>& lidarFrameList);
	void EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
						   const Eigen::Matrix4d& exTlb,
						   const Eigen::Vector3d& gravity,
						   nav_msgs::Odometry& debugInfo);
	void Estimate(std::list<LidarFrame>& lidarFrameList,
				  const Eigen::Matrix4d& exTlb,
				  const Eigen::Vector3d& gravity);

	pcl::PointCloud<PointType>::Ptr get_corner_map(){
		return map_manager->get_corner_map();
	}
	pcl::PointCloud<PointType>::Ptr get_surf_map(){
		return map_manager->get_surf_map();
	}
	pcl::PointCloud<PointType>::Ptr get_nonfeature_map(){
		return map_manager->get_nonfeature_map();
	}
	void MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
						   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
						   const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
						   const Eigen::Matrix4d& transformTobeMapped);

    bool run_imu_align(std::deque<LidarFrame>& frames,
                       Eigen::Vector3d& g_b,
                       const Eigen::Matrix3d& ex_r_lb, 
                       const Eigen::Vector3d& ex_t_bl,
                       const size_t& win_size);

private:
	/** \brief store map points */
	MAP_MANAGER* map_manager;
	ImuAligner* imu_aligner_ptr_

	double para_PR[silde_windows_size][6];
	double para_VBias[silde_windows_size][9];
	MarginalizationInfo *last_marginalization_info = nullptr;
	std::vector<double *> last_marginalization_parameter_blocks;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureLast;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromLocal;
	pcl::PointCloud<PointType>::Ptr laserCloudCornerForMap;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfForMap;
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureForMap;
	Eigen::Matrix4d transformForMap;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerStack;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfStack;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureStack;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeNonFeatureFromLocal;
	pcl::VoxelGrid<PointType> downSizeFilterCorner;
	pcl::VoxelGrid<PointType> downSizeFilterSurf;
	pcl::VoxelGrid<PointType> downSizeFilterNonFeature;
	std::mutex mtx_Map;
	std::thread threadMap;

	pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
	pcl::KdTreeFLANN<PointType> SurfKdMap[10000];
	pcl::KdTreeFLANN<PointType> NonFeatureKdMap[10000];

	pcl::PointCloud<PointType> GlobalSurfMap[10000];
	pcl::PointCloud<PointType> GlobalCornerMap[10000];
	pcl::PointCloud<PointType> GlobalNonFeatureMap[10000];

	int laserCenWidth_last = 10;
	int laserCenHeight_last = 5;
	int laserCenDepth_last = 10;

	static const int localMapWindowSize = 50;
	int localMapID = 0;
	pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
	pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
	pcl::PointCloud<PointType>::Ptr localNonFeatureMap[localMapWindowSize];

	int map_update_ID = 0;

	int map_skip_frame = 2; //every map_skip_frame frame update map
	double plan_weight_tan = 0.0;
	double thres_dist = 1.0;
};
} // end of namespace

#endif // LIO_POSE_ESTIMATOR_HPP
