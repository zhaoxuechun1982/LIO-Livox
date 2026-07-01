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
#include "lio/pose_estimator.hpp"
#include "lio/map_manager.hpp"
#include "utils/ceres_utils.hpp"
#include "sophus/so3.hpp"
#include <ros/ros.h>

namespace lio
{
/*
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
*/

void LidarFeatureMatcher::match_point_to_line(CeresCostFunctionPtrVector& edges,
                                              FeatureLineVector& line_features,
                                              const PointCloudTypePtr& corner_cloud,
                                              const PointCloudTypePtr& cloud_corner_local,
                                              const PointKdTreeTypePtr& kdtree_local,
                                              const Eigen::Matrix4d& exTlb,
                                              const Eigen::Matrix4d& m4d)
{
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  
  if(!line_features.empty()){
    for(const auto& l : line_features){
      auto* e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                               l.lineP1,
                                               l.lineP2,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();

  int laserCloudCornerStackNum = corner_cloud->points.size();
  PointCloudTypePtr kd_pointcloud(new PointCloudType);
  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  for (int i = 0; i < laserCloudCornerStackNum; i++) {
    _pointOri = corner_cloud->points[i];
    MapManager::point_associate_to_map(&_pointOri, &_pointSel, m4d);
    int id = host_->map_manager_ptr_->FindUsedCornerMap(&_pointSel,
                                                        host_->laser_center_width_last_,
                                                        host_->laser_center_height_last_,
                                                        host_->laser_center_depth_last_);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(host_->global_corner_map_[id].points.size() > 100) {
      host_->kdtree_corner_map_[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);
      
      if (_pointSearchSqDis[4] < host_->thres_dist) {

        debug_num1 ++;
      float cx = 0;
      float cy = 0;
      float cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += host_->global_corner_map_[id].points[_pointSearchInd[j]].x;
        cy += host_->global_corner_map_[id].points[_pointSearchInd[j]].y;
        cz += host_->global_corner_map_[id].points[_pointSearchInd[j]].z;
      }
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0;
      float a12 = 0;
      float a13 = 0;
      float a22 = 0;
      float a23 = 0;
      float a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = host_->global_corner_map_[id].points[_pointSearchInd[j]].x - cx;
        float ay = host_->global_corner_map_[id].points[_pointSearchInd[j]].y - cy;
        float az = host_->global_corner_map_[id].points[_pointSearchInd[j]].z - cz;

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      _matA1(0, 0) = a11;
      _matA1(0, 1) = a12;
      _matA1(0, 2) = a13;
      _matA1(1, 0) = a12;
      _matA1(1, 1) = a22;
      _matA1(1, 2) = a23;
      _matA1(2, 0) = a13;
      _matA1(2, 1) = a23;
      _matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

      if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
        debug_num12 ++;
        float x1 = cx + 0.1 * unit_direction[0];
        float y1 = cy + 0.1 * unit_direction[1];
        float z1 = cz + 0.1 * unit_direction[2];
        float x2 = cx - 0.1 * unit_direction[0];
        float y2 = cy - 0.1 * unit_direction[1];
        float z2 = cz - 0.1 * unit_direction[2];

        Eigen::Vector3d tripod1(x1, y1, z1);
        Eigen::Vector3d tripod2(x2, y2, z2);
        auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                 tripod1,
                                                 tripod2,
                                                 Tbl,
                                                 Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
        edges.push_back(e);
        line_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                   tripod1,
                                   tripod2);
        line_features.back().calculate_error(m4d);

        continue;
      }
    }
    
    }

    if(cloud_corner_local->points.size() > 20 ){
      kdtree_local->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < host_->thres_dist) {

        debug_num2 ++;
        float cx = 0;
        float cy = 0;
        float cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += cloud_corner_local->points[_pointSearchInd2[j]].x;
          cy += cloud_corner_local->points[_pointSearchInd2[j]].y;
          cz += cloud_corner_local->points[_pointSearchInd2[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax = cloud_corner_local->points[_pointSearchInd2[j]].x - cx;
          float ay = cloud_corner_local->points[_pointSearchInd2[j]].y - cy;
          float az = cloud_corner_local->points[_pointSearchInd2[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        _matA1(0, 0) = a11;
        _matA1(0, 1) = a12;
        _matA1(0, 2) = a13;
        _matA1(1, 0) = a12;
        _matA1(1, 1) = a22;
        _matA1(1, 2) = a23;
        _matA1(2, 0) = a13;
        _matA1(2, 1) = a23;
        _matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
          debug_num22++;
          float x1 = cx + 0.1 * unit_direction[0];
          float y1 = cy + 0.1 * unit_direction[1];
          float z1 = cz + 0.1 * unit_direction[2];
          float x2 = cx - 0.1 * unit_direction[0];
          float y2 = cy - 0.1 * unit_direction[1];
          float z2 = cz - 0.1 * unit_direction[2];

          Eigen::Vector3d tripod1(x1, y1, z1);
          Eigen::Vector3d tripod2(x2, y2, z2);
          auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  tripod1,
                                                  tripod2,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          line_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    tripod1,
                                    tripod2);
          line_features.back().calculate_error(m4d);
        }
      }
    } 
  }
}

void LidarFeatureMatcher::match_point_to_plane(CeresCostFunctionPtrVector& edges,
                                   FeaturePlaneVector& plane_features,
                                   const PointCloudTypePtr& cloud_surface,
                                   const PointCloudTypePtr& cloud_surface_local,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtree_local,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d)
{
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  if(!plane_features.empty()){
    for(const auto& p : plane_features){
      auto* e = Cost_NavState_IMU_Plan::Create(p.pointOri,
                                               p.pa,
                                               p.pb,
                                               p.pc,
                                               p.pd,
                                               Tbl,
                                               Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  int laserCloudSurfStackNum = cloud_surface->points.size();

  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    _pointOri = cloud_surface->points[i];
    MapManager::point_associate_to_map(&_pointOri, &_pointSel, m4d);

    int id = host_->map_manager_ptr_->FindUsedSurfMap(&_pointSel,host_->laser_center_width_last_,host_->laser_center_height_last_,host_->laser_center_depth_last_);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(host_->global_surface_map_[id].points.size() > 50) {
      host_->kdtree_surface_map_[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      if (_pointSearchSqDis[4] < 1.0) {
        debug_num1 ++;
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = host_->global_surface_map_[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = host_->global_surface_map_[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = host_->global_surface_map_[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * host_->global_surface_map_[id].points[_pointSearchInd[j]].x +
                        pb * host_->global_surface_map_[id].points[_pointSearchInd[j]].y +
                        pc * host_->global_surface_map_[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          debug_num12 ++;
          auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                  pa,
                                                  pb,
                                                  pc,
                                                  pd,
                                                  Tbl,
                                                  Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          plane_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          plane_features.back().calculate_error(m4d);

          continue;
        }
        
      }
    }
    if(cloud_surface_local->points.size() > 20 ){
    kdtree_local->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
    if (_pointSearchSqDis2[4] < 1.0) {
      debug_num2++;
      for (int j = 0; j < 5; j++) { 
        _matA0(j, 0) = cloud_surface_local->points[_pointSearchInd2[j]].x;
        _matA0(j, 1) = cloud_surface_local->points[_pointSearchInd2[j]].y;
        _matA0(j, 2) = cloud_surface_local->points[_pointSearchInd2[j]].z;
      }
      _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

      float pa = _matX0(0, 0);
      float pb = _matX0(1, 0);
      float pc = _matX0(2, 0);
      float pd = 1;

      float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (std::fabs(pa * cloud_surface_local->points[_pointSearchInd2[j]].x +
                      pb * cloud_surface_local->points[_pointSearchInd2[j]].y +
                      pc * cloud_surface_local->points[_pointSearchInd2[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        debug_num22 ++;
        auto* e = Cost_NavState_IMU_Plan::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
        edges.push_back(e);
        plane_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                  pa,
                                  pb,
                                  pc,
                                  pd);
        plane_features.back().calculate_error(m4d);
      }
    }
  }

  }

}

void LidarFeatureMatcher::match_point_to_plane_vector(CeresCostFunctionPtrVector& edges,
                                   FeaturePlaneVectorVector& plane_vector_features,
                                   const PointCloudTypePtr& cloud_surface,
                                   const PointCloudTypePtr& cloud_surface_local,
                                   const PointKdTreeTypePtr& kdtree_local,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d)
{
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  if(!plane_vector_features.empty()){
    for(const auto& p : plane_vector_features){
      auto* e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                   p.pointProj,
                                                   Tbl,
                                                   p.sqrt_info);
      edges.push_back(e);
    }
    return;
  }
  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();
  int laserCloudSurfStackNum = cloud_surface->points.size();

  int debug_num1 = 0;
  int debug_num2 = 0;
  int debug_num12 = 0;
  int debug_num22 = 0;
  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    _pointOri = cloud_surface->points[i];
    MapManager::point_associate_to_map(&_pointOri, &_pointSel, m4d);

    int id = host_->map_manager_ptr_->FindUsedSurfMap(&_pointSel,host_->laser_center_width_last_,host_->laser_center_height_last_,host_->laser_center_depth_last_);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(host_->global_surface_map_[id].points.size() > 50) {
      host_->kdtree_surface_map_[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

      if (_pointSearchSqDis[4] < host_->thres_dist) {
        debug_num1 ++;
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = host_->global_surface_map_[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = host_->global_surface_map_[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = host_->global_surface_map_[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * host_->global_surface_map_[id].points[_pointSearchInd[j]].x +
                        pb * host_->global_surface_map_[id].points[_pointSearchInd[j]].y +
                        pc * host_->global_surface_map_[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          debug_num12 ++;
          double dist = pa * _pointSel.x +
                        pb * _pointSel.y +
                        pc * _pointSel.z + pd;
          Eigen::Vector3d omega(pa, pb, pc);
          Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
          Eigen::Vector3d e1(1, 0, 0);
          Eigen::Matrix3d J = e1 * omega.transpose();
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
          Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
          Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
          info(1, 1) *= host_->plan_weight_tan;
          info(2, 2) *= host_->plan_weight_tan;
          Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

          auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                       point_proj,
                                                       Tbl,
                                                       sqrt_info);
          edges.push_back(e);
          plane_vector_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                     point_proj,
                                     sqrt_info);
          plane_vector_features.back().calculate_error(m4d);

          continue;
        }
        
      }
    }


    if(cloud_surface_local->points.size() > 20 ) {
    kdtree_local->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
    if (_pointSearchSqDis2[4] < host_->thres_dist) {
      debug_num2++;
      for (int j = 0; j < 5; j++) { 
        _matA0(j, 0) = cloud_surface_local->points[_pointSearchInd2[j]].x;
        _matA0(j, 1) = cloud_surface_local->points[_pointSearchInd2[j]].y;
        _matA0(j, 2) = cloud_surface_local->points[_pointSearchInd2[j]].z;
      }
      _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

      float pa = _matX0(0, 0);
      float pb = _matX0(1, 0);
      float pc = _matX0(2, 0);
      float pd = 1;

      float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (std::fabs(pa * cloud_surface_local->points[_pointSearchInd2[j]].x +
                      pb * cloud_surface_local->points[_pointSearchInd2[j]].y +
                      pc * cloud_surface_local->points[_pointSearchInd2[j]].z + pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        debug_num22 ++;
        double dist = pa * _pointSel.x +
                      pb * _pointSel.y +
                      pc * _pointSel.z + pd;
        Eigen::Vector3d omega(pa, pb, pc);
        Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
        Eigen::Vector3d e1(1, 0, 0);
        Eigen::Matrix3d J = e1 * omega.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
        Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
        info(1, 1) *= host_->plan_weight_tan;
        info(2, 2) *= host_->plan_weight_tan;
        Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

        auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                      point_proj,
                                                      Tbl,
                                                      sqrt_info);
        edges.push_back(e);
        plane_vector_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    point_proj,
                                    sqrt_info);
        plane_vector_features.back().calculate_error(m4d);
      }
    }
    }
  }
}


void LidarFeatureMatcher::match_none_feature_icp(CeresCostFunctionPtrVector& edges,
                                                 FeatureNoneVector& none_features,
                                                 const PointCloudTypePtr& cloud_none,
                                                 const PointCloudTypePtr& cloud_none_local,
                                                 const PointKdTreeTypePtr& kdtree_local,
                                                 const Eigen::Matrix4d& exTlb,
                                                 const Eigen::Matrix4d& m4d)
{
  Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
  Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
  Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
  if(!none_features.empty()){
    for(const auto& p : none_features){
      auto* e = Cost_NonFeature_ICP::Create(p.pointOri,
                                            p.pa,
                                            p.pb,
                                            p.pc,
                                            p.pd,
                                            Tbl,
                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
      edges.push_back(e);
    }
    return;
  }

  PointType _pointOri, _pointSel, _coeff;
  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;
  std::vector<int> _pointSearchInd2;
  std::vector<float> _pointSearchSqDis2;

  Eigen::Matrix< double, 5, 3 > _matA0;
  _matA0.setZero();
  Eigen::Matrix< double, 5, 1 > _matB0;
  _matB0.setOnes();
  _matB0 *= -1;
  Eigen::Matrix< double, 3, 1 > _matX0;
  _matX0.setZero();

  int laserCloudNonFeatureStackNum = cloud_none->points.size();
  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    _pointOri = cloud_none->points[i];
    MapManager::point_associate_to_map(&_pointOri, &_pointSel, m4d);
    int id = host_->map_manager_ptr_->FindUsedNonFeatureMap(&_pointSel,host_->laser_center_width_last_,host_->laser_center_height_last_,host_->laser_center_depth_last_);

    if(id == 5000) continue;

    if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

    if(host_->global_none_map_[id].points.size() > 100) {
      host_->kdtree_none_map_[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);
      if (_pointSearchSqDis[4] < 1 * host_->thres_dist) {
        for (int j = 0; j < 5; j++) {
          _matA0(j, 0) = host_->global_none_map_[id].points[_pointSearchInd[j]].x;
          _matA0(j, 1) = host_->global_none_map_[id].points[_pointSearchInd[j]].y;
          _matA0(j, 2) = host_->global_none_map_[id].points[_pointSearchInd[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * host_->global_none_map_[id].points[_pointSearchInd[j]].x +
                        pb * host_->global_none_map_[id].points[_pointSearchInd[j]].y +
                        pc * host_->global_none_map_[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if(planeValid) {

          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          none_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          none_features.back().calculate_error(m4d);

          continue;
        }
      }
    
    }

    if(cloud_none_local->points.size() > 20 ){
      kdtree_local->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
      if (_pointSearchSqDis2[4] < 1 * host_->thres_dist) {
        for (int j = 0; j < 5; j++) { 
          _matA0(j, 0) = cloud_none_local->points[_pointSearchInd2[j]].x;
          _matA0(j, 1) = cloud_none_local->points[_pointSearchInd2[j]].y;
          _matA0(j, 2) = cloud_none_local->points[_pointSearchInd2[j]].z;
        }
        _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

        float pa = _matX0(0, 0);
        float pb = _matX0(1, 0);
        float pc = _matX0(2, 0);
        float pd = 1;

        float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (std::fabs(pa * cloud_none_local->points[_pointSearchInd2[j]].x +
                        pb * cloud_none_local->points[_pointSearchInd2[j]].y +
                        pc * cloud_none_local->points[_pointSearchInd2[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if(planeValid) {

          auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
          edges.push_back(e);
          none_features.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                    pa,
                                    pb,
                                    pc,
                                    pd);
          none_features.back().calculate_error(m4d);
        }
      }
    }
  }

}


}