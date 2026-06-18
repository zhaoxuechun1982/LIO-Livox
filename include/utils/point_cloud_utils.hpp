#ifndef POINT_CLOUD_UTILS_HPP
#define POINT_CLOUD_UTILS_HPP

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <vector>

namespace lidar_proc
{
using PointXYZ = pcl::PointXYZ;
using PointXYZCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointXYZCloudPtr = PointXYZCloud::Ptr;
using KdTreePtr = pcl::KdTreeFLANN<pcl::PointXYZ>;

// Constant definition for PCA neighbor calculation
// Minimum valid neighbor point count to run PCA calculation
constexpr int PCA_MIN_NEIGHBOR_COUNT = 5;
// Epsilon to avoid division by zero when normalizing eigen values
constexpr float DIVISION_EPSILON = 1e-6f;

/**
 * @brief Local plane PCA result structure
 */
struct NeighborPlanePCA
{
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector3f eigen_values;
  std::vector<int> neighbor_indices;

  NeighborPlanePCA(void) {
    eigen_vectors.setZero();
    eigen_values.setZero();
    neighbor_indices.clear();
  }
};

/**
 * @brief Static point cloud geometry tool class, no instance allowed
 */
class PointCloudUtils
{
public:
  // Disable constructor & destructor, pure static utility class
  PointCloudUtils() = delete;
  ~PointCloudUtils() = delete;

  /**
   * @brief Calculate local plane PCA feature for target search point
   * @param out_npp Output PCA result struct
   * @param cloud Input full point cloud
   * @param kdtree Pre-built KDTree of cloud
   * @param search_point Target query point
   * @param search_radius Radius for neighborhood search (m)
   * @return Valid neighbor count; 0 if neighbor count insufficient or solve failed
   */
  static int calculate_neighbor_plane_pca(NeighborPlanePCA& out_npp,
                                          PointXYZCloudPtr cloud_ptr,
                                          KdTreePtr kdtree,
                                          PointXYZ search_point,
                                          float search_radius);

  /**
   * @brief Calculate 3x3 rotation matrix via Rodrigues formula with two input vectors
   * @param[out] rtm Output row-major 3*3 matrix array, length = 9
   * @param[inout] v0 Input vector 1, will be normalized & modified inside function
   * @param[inout] v1 Input vector 2, will be normalized & modified inside function
   * @return Fixed return 0
   * @warning Original v0 / v1 float array will be overwritten
   */
  static int calculate_rotation_transformation_matrix(float* rtm, float* v0, float* v1);
};

} // namespace lidar_proc

#endif // POINT_CLOUD_UTILS_HPP