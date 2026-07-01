#ifndef LIO_POINT_CLOUD_TYPE_HPP
#define LIO_POINT_CLOUD_TYPE_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace lio_data_type
{
  using PointType = pcl::PointXYZINormal;
  using PointCloudType = pcl::PointCloud<PointType>;
  using PointCloudTypePtr = PointCloudType::Ptr;
  using PointKdTreeType = pcl::KdTreeFLANN<PointType>;
  using PointKdTreeTypePtr= PointKdTreeType::Ptr;
  using PointVoxelGridType = pcl::VoxelGrid<PointType>;
} // end of namespace lio_data_type 

#endif // end of LIO_POINT_CLOUD_TYPE_HPP