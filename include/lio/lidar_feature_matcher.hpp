#ifndef LIO_LIDAR_FEATURE_MATCHER_HPP
#define LIO_LIDAR_FEATURE_MATCHER_HPP

#include "type/lidar_feature.hpp"
#include <vector>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>

//#include "lio/estimator_data.hpp"
//#include "type/point_type.hpp"

namespace lio
{
class LidarFeatureMatcher
{
public:
  static LidarFeatureMatcher& instance(void);
  static LidarFeatureMatcher* instance_pointer(void);
  static void destroy_instance(void);

  void match_point_to_line(std::vector<ceres::CostFunction*>& edges,
                           std::vector<FeatureLine>& line_features,
                           const pcl::PointCloud<PointType>::Ptr& corner_cloud,
                           const pcl::PointCloud<PointType>::Ptr& local_corner_map,
                           const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                           const Eigen::Matrix4d& T_lb,
                           const Eigen::Matrix4d& pose_map,
                           const EstimatorData& data);

    void match_point_to_plane(std::vector<ceres::CostFunction*>& edges,
                              std::vector<FeaturePlane>& plane_features,
                              const pcl::PointCloud<PointType>::Ptr& surf_cloud,
                              const pcl::PointCloud<PointType>::Ptr& local_surf_map,
                              const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                              const Eigen::Matrix4d& T_lb,
                              const Eigen::Matrix4d& pose_map,
                              const EstimatorData& data);

    void match_point_to_plane_vector(std::vector<ceres::CostFunction*>& edges,
                                     std::vector<FeaturePlaneVector>& plane_vec_features,
                                     const pcl::PointCloud<PointType>::Ptr& surf_cloud,
                                     const pcl::PointCloud<PointType>::Ptr& local_surf_map,
                                     const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                     const Eigen::Matrix4d& T_lb,
                                     const Eigen::Matrix4d& pose_map,
                                     const EstimatorData& data);

    void match_non_feature_plane(std::vector<ceres::CostFunction*>& edges,
                                  std::vector<FeatureNon>& non_features,
                                  const pcl::PointCloud<PointType>::Ptr& non_cloud,
                                  const pcl::PointCloud<PointType>::Ptr& local_non_map,
                                  const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                  const Eigen::Matrix4d& T_lb,
                                  const Eigen::Matrix4d& pose_map,
                                  const EstimatorData& data);

private:
  LidarFeatureMatcher(); 
  ~LidarFeatureMatcher() = default;
  LidarFeatureMatcher(const LidarFeatureMatcher&) = delete;
  LidarFeatureMatcher& operator=(const LidarFeatureMatcher&) = delete;

  static LidarFeatureMatcher* instance_ptr_;
};

} // end of namespace lio

#endif // end of LIO_LIDAR_FEATURE_MATCHER_HPP