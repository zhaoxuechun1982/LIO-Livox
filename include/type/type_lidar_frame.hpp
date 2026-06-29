#ifndef LIO_TYPE_LIDAR_FRAME_HPP
#define LIO_TYPE_LIDAR_FRAME_HPP

#include "type/type_point_cloud.hpp"
#include "lio/imu_integrator.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lio_data_type
{
  /** @brief lidar frame struct 
   */
  struct LidarFrame
  {
	PointCloudTypePtr laser_cloud;
	IMUIntegrator imu_integrator;
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	Eigen::Quaterniond q;
	Eigen::Vector3d bg;
	Eigen::Vector3d ba;
	double timestamp;

	LidarFrame(void):
	  p(Eigen::Vector3d::Zero()),
	  v(Eigen::Vector3d::Zero()),
	  q(Eigen::Quaterniond::Identity()),
	  bg(Eigen::Vector3d::Zero()),
	  ba(Eigen::Vector3d::Zero())
    {}
  };
} // end of namespace lio_data_type

#endif // end of LIO_TYPE_LIDAR_FRAME_HPP