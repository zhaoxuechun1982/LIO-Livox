#ifndef LIO_LIDAR_FRAME_HPP
#define LIO_LIDAR_FRAME_HPP

#include "lio/imu_integrator.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lio 
{
  using PointType = pcl::PointXYZINormal;
  
  /** @brief lidar frame struct 
   * 
   */
  struct LidarFrame
  {
	pcl::PointCloud<PointType>::Ptr laser_cloud;
	IMUIntegrator imu_integrator;
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	Eigen::Quaterniond q;
	Eigen::Vector3d bg;
	Eigen::Vector3d ba;
	double timestamp;

	LidarFrame(void)
    {
	  p.setZero();
	  v.setZero();
	  q.setIdentity();
	  bg.setZero();
	  ba.setZero();
	  timestamp = 0.0;
	}
  };
}

#endif