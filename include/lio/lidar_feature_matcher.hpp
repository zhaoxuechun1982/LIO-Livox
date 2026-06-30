#ifndef LIO_LIDAR_FEATURE_MATCHER_HPP
#define LIO_LIDAR_FEATURE_MATCHER_HPP

#include "type/ceres_type.hpp"
#include "type/point_cloud_type.hpp"
#include "type/lidar_feature_type.hpp"
//#include "lio/estimator_data.hpp"
#include <Eigen/Eigen>
#include <vector>

using namespace lio_data_type;
namespace lio
{
class LidarFeatureMatcher
{
public:
  static LidarFeatureMatcher& instance(void);
  static LidarFeatureMatcher* instance_pointer(void);
  static void destroy_instance(void);

  void match_point_to_line(CeresCostFunctionPtrVector& edges,
                           FeatureLineVector& line_features,
                           const PointCloudTypePtr& corner_cloud,
                           const PointCloudTypePtr& local_corner_map,
                           const PointKdTreeTypePtr& local_kdtree,
                           const Eigen::Matrix4d& T_lb,
                           const Eigen::Matrix4d& pose_map,
                           const EstimatorData& data);

    void match_point_to_plane(CeresCostFunctionPtrVector& edges,
                              FeaturePlaneVector& plane_features,
                              const PointCloudTypePtr& surf_cloud,
                              const PointCloudTypePtr& local_surf_map,
                              const PointKdTreeTypePtr& local_kdtree,
                              const Eigen::Matrix4d& T_lb,
                              const Eigen::Matrix4d& pose_map,
                              const EstimatorData& data);

    void match_point_to_plane_vector(CeresCostFunctionPtrVector& edges,
                                     FeaturePlaneVectorVector& plane_vec_features,
                                     const PointCloudTypePtr& surf_cloud,
                                     const PointCloudTypePtr& local_surf_map,
                                     const PointKdTreeTypePtr& local_kdtree,
                                     const Eigen::Matrix4d& T_lb,
                                     const Eigen::Matrix4d& pose_map,
                                     const EstimatorData& data);

    void match_non_feature_plane(CeresCostFunctionPtrVector& edges,
                                  FeatureNoneVector& non_features,
                                  const PointCloudTypePtr& non_cloud,
                                  const PointCloudTypePtr& local_non_map,
                                  const PointKdTreeTypePtr& local_kdtree,
                                  const Eigen::Matrix4d& T_lb,
                                  const Eigen::Matrix4d& pose_map,
                                  const EstimatorData& data);

private:
  LidarFeatureMatcher() = default; 
  ~LidarFeatureMatcher() = default;
  LidarFeatureMatcher(const LidarFeatureMatcher&) = delete;
  LidarFeatureMatcher& operator=(const LidarFeatureMatcher&) = delete;

  static LidarFeatureMatcher* instance_ptr_;
};

} // end of namespace lio

#endif // end of LIO_LIDAR_FEATURE_MATCHER_HPP