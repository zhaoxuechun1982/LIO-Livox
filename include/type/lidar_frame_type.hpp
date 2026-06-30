#ifndef LIO_LIDAR_FRAME_TYPE_HPP
#define LIO_LIDAR_FRAME_TYPE_HPP

#include "type/point_cloud_type.hpp"
#include "lio/imu_integrator.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <list>
#include <deque>

namespace lio_data_type
{
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

  using LidarFrameVector = std::vector<LidarFrame>;
  using LidarFrameList = std::list<LidarFrame>;
  using LidarFrameDeque = std::deque<LidarFrame>;

} // end of namespace lio_data_type

#endif // end of LIO_LIDAR_FRAME_TYPE_HPP