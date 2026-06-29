/*
 * @FileName: lidar_feature_matcher.cpp
 * @Brief: IMU coarse initial alignment (attitude initialization)
 * @Description: Use accelerometer & gravity vector to compute initial roll/pitch,
 *               Use magnetometer or angular velocity to estimate initial yaw.
 *               This module provides gravity-based coarse alignment for INS.
 * @Author: Jimmy Zhao
 * @Date: 2026-06-24
 */

#include "lio/lidar_feature_matcher.hpp"
#include "utils/ceres_utils.hpp"
#include "sophus/so3.hpp"
#include <ros/ros.h>

namespace lio
{

LidarFeatureMatcher* LidarFeatureMatcher::instance_ptr_ = nullptr;

LidarFeatureMatcher* LidarFeatureMatcher::instance_pointer(void)
{
  if (instance_ptr_ == nullptr) {
    instance_ptr_ = new LidarFeatureMatcher();
  }
  
  return instance_ptr_;
}

LidarFeatureMatcher& LidarFeatureMatcher::instance(void)
{
  return *instance_pointer();
}

void LidarFeatureMatcher::destroy_instance(void)
{
  if (instance_ptr_ != nullptr) {
    delete instance_ptr_;
    instance_ptr_ = nullptr;
  }
}

void LidarFeatureMatcher::match_point_to_line(std::vector<ceres::CostFunction*>& edges,
                                         std::vector<FeatureLine>& line_features,
                                         const pcl::PointCloud<PointType>::Ptr& corner_cloud,
                                         const pcl::PointCloud<PointType>::Ptr& local_corner_map,
                                         const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                         const Eigen::Matrix4d& T_lb,
                                         const Eigen::Matrix4d& pose_map,
                                         const EstimatorData& data)
{

}

void LidarFeatureMatcher::match_point_to_plane(std::vector<ceres::CostFunction*>& edges,
                                          std::vector<FeaturePlane>& plane_features,
                                          const pcl::PointCloud<PointType>::Ptr& surf_cloud,
                                          const pcl::PointCloud<PointType>::Ptr& local_surf_map,
                                          const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                          const Eigen::Matrix4d& T_lb,
                                          const Eigen::Matrix4d& pose_map,
                                          const EstimatorData& data)
{

}

void LidarFeatureMatcher::match_point_to_plane_vector(std::vector<ceres::CostFunction*>& edges,
                                                 std::vector<FeaturePlaneVector>& plane_vec_features,
                                                 const pcl::PointCloud<PointType>::Ptr& surf_cloud,
                                                 const pcl::PointCloud<PointType>::Ptr& local_surf_map,
                                                 const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                                 const Eigen::Matrix4d& T_lb,
                                                 const Eigen::Matrix4d& pose_map,
                                                 const EstimatorData& data)
{

}

void LidarFeatureMatcher::match_non_feature_plane(std::vector<ceres::CostFunction*>& edges,
                                            std::vector<FeatureNon>& non_features,
                                            const pcl::PointCloud<PointType>::Ptr& non_cloud,
                                            const pcl::PointCloud<PointType>::Ptr& local_non_map,
                                            const pcl::KdTreeFLANN<PointType>::Ptr& local_kdtree,
                                            const Eigen::Matrix4d& T_lb,
                                            const Eigen::Matrix4d& pose_map,
                                            const EstimatorData& data)
{

}


}