#ifndef LIO_TYPE_POINT_CLOUD_HPP
#define LIO_TYPE_POINT_CLOUD_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lio_data_type
{
  using PointType = pcl::PointXYZINormal;
  using PointCloudType = pcl::PointCloud<PointType>;
  using PointCloudTypePtr = PointCloudType::Ptr;
} // end of namespace lio_data_type 

#endif // end of LIO_TYPE_POINT_CLOUD_HPP