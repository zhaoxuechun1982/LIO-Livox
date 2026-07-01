#ifndef LIO_LIDAR_FEATURE_MATCHER_HPP
#define LIO_LIDAR_FEATURE_MATCHER_HPP

#include "type/ceres_type.hpp"
#include "type/point_cloud_type.hpp"
#include "type/lidar_feature_type.hpp"
#include <Eigen/Eigen>
#include <vector>

using namespace lio_data_type;
namespace lio
{
class PoseEstimator; 

class LidarFeatureMatcher
{

public:
  // static LidarFeatureMatcher& instance(void);
  // static LidarFeatureMatcher* instance_pointer(void);
  // static void destroy_instance(void);
  explicit LidarFeatureMatcher(PoseEstimator* host) : host_(host) {}
  ~LidarFeatureMatcher() = default;

  void match_point_to_line(CeresCostFunctionPtrVector& edges,
                           FeatureLineVector& line_features,
                           const PointCloudTypePtr& corner_cloud,
                           const PointCloudTypePtr& cloud_corner_local,
                           const PointKdTreeTypePtr& kdtree_local,
                           const Eigen::Matrix4d& exTlb,
                           const Eigen::Matrix4d& m4d);

  void match_point_to_plane(CeresCostFunctionPtrVector& edges,
                            FeaturePlaneVector& plane_features,
                            const PointCloudTypePtr& cloud_surface,
                            const PointCloudTypePtr& cloud_surface_local,
                            const PointKdTreeTypePtr& kdtree_local,
                            const Eigen::Matrix4d& exTlb,
                            const Eigen::Matrix4d& m4d);

  void match_point_to_plane_vector(CeresCostFunctionPtrVector& edges,
                                   FeaturePlaneVectorVector& plane_vector_features,
                                   const PointCloudTypePtr& cloud_surface,
                                   const PointCloudTypePtr& cloud_surface_local,
                                   const PointKdTreeTypePtr& kdtree_local,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d);

  void match_none_feature_icp(CeresCostFunctionPtrVector& edges,
                              FeatureNoneVector& none_features,
                              const PointCloudTypePtr& cloud_none,
                              const PointCloudTypePtr& cloud_none_local,
                              const PointKdTreeTypePtr& kdtree_local,
                              const Eigen::Matrix4d& exTlb,
                              const Eigen::Matrix4d& m4d);

private:
  //LidarFeatureMatcher() = default; 
  LidarFeatureMatcher(const LidarFeatureMatcher&) = delete;
  LidarFeatureMatcher& operator=(const LidarFeatureMatcher&) = delete;

  //static LidarFeatureMatcher* instance_ptr_;
  PoseEstimator* host_ = nullptr;
};
} // end of namespace lio

#endif // end of LIO_LIDAR_FEATURE_MATCHER_HPP
