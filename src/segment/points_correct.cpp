#include "utils/point_cloud_utils.hpp"
#include "segment/pointsCorrect.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>

namespace lidar_proc
{
// Initialize static persistent state
float PointsCorrect::m_ground_pos[6] = {0.0f};
int PointsCorrect::m_frame_count = 0;

void PointsCorrect::reset(void)
{
  std::memset(m_gound_pos, 0, 6*sizeof(float));
  m_frame_count = 0;
}

int PointsCorrect::estimate_and_smooth_ground(float* pos_out, float* points_in, int points_num)
{
  // Null pointer safety guard
  if (pos_out == nullptr || points_in == nullptr)
  {
    return -1;
  }

  // Allocate temp buffer for ground candidate points (max 60000 points, ~0.96MB)
  float *points_buffer = (float*)calloc(MAX_GND_CANDIDATE_NUM*4, sizeof(float));

  // Filter ground points
  int temp_points_num = filter_ground_candidate(points_buffer, points_in, points_num);
  if (temp_points_num < 3) {
    std::cout << "Too few ground points!" << std::endl;
  }

  // Calculate current frame ground plane feature
  float temp_pos[6] = {0.0f};
  int valid_ground_patch_count = calculate_ground_pos(temp_pos, points_buffer, temp_points_num, 1.0); 
  free(points_buffer);

  /*
  Multi-frame temporal smoothing for ground plane parameters, empirical tuning logic.
  Limitation: This is a scene-specific engineering compromise without rigorous mathematical model.
  Rule overview:
  1. Empty historical cache: directly initialize with current frame ground.
  2. Valid historical ground exists:
   - Unstable frame counter not reach threshold + current frame valid ground:
     Small XY normal difference: average old & new ground to suppress jitter, reset counter;
     Large normal difference: only accumulate unstable frame counter, no ground update.
   - Counter exceeds threshold: abandon historical data, fully overwrite with current ground and reset counter.
  Defects: Only judge normal XY deviation, fixed 0.5 smoothing weight, hard frame threshold switch.
  */
  // Temporal smoothing logic for ground plane
  // 1. No historical ground data: initialize cache with current frame ground
  // 2. Existing valid ground cache:
  //    a. Within frame tolerance & current frame valid:
  //       - Small normal deviation: average old & new ground to suppress jitter, reset counter
  //       - Large deviation: only increase unstable frame counter
  //    b. Counter exceeds threshold: discard old cache, fully replace with new ground and reset counter
  if (m_gound_pos[5] == 0.0f) {
    // No historical ground data, use current frame directly
    memcpy(m_gound_pos, temp_pos, 6*sizeof(float));
  }
  else {  
    if (m_frame_count < FRAME_COUNT_THRESHOLD && temp_pos[5] != 0.0f) {
      float diff_abs_x = std::fabs(m_gnd_pos[0] - frame_ground_param[0]);
      float diff_abs_y = std::fabs(m_gnd_pos[1] - frame_ground_param[1]);

      if (valid_ground_patch_count > 0 
       && diff_abs_x < NORMAL_DIFF_THR  
       && diff_abs_y < NORMAL_DIFF_THR) {
        // Temporal average smooth for stable ground
        for(int i = 0; i < 6; i++) {
          m_gound_pos[i] = (m_gound_pos[i] + temp_pos[i]) * 0.5;
        }
        m_frame_count = 0;
      }
      else {
        m_frame_count++;
      }
    }
    else if(temp_pos[5] != 0) {
      // Overwrite history when frame count exceeds threshold
      memcpy(m_gound_pos, temp_pos, 6*sizeof(float));
      m_frame_count = 0;
    }
  }
   
  memcpy(pos_out, m_gound_pos, 6*sizeof(float));

  return 0;
}

int PointsCorrect::filter_ground_candidate(float* points_out, float* points_in, int points_num)
{
  // ... original implementation
  int outNum = 0;
  float dx = 2;
  float dy = 2;
  int x_len = 20;
  int y_len = 10;
  int nx = 2 * x_len / dx; // 20
  int ny = 2 * y_len / dy; // 10
  float offx=-20,offy=-10;
  float THR = 0.4;
  
  float* imgMinZ  = (float*)calloc(nx*ny, sizeof(float));
  float* imgMaxZ  = (float*)calloc(nx*ny, sizeof(float));
  float* imgSumZ  = (float*)calloc(nx*ny, sizeof(float));
  float* imgMeanZ = (float*)calloc(nx*ny, sizeof(float));
  int* imgNumZ = (int*)calloc(nx*ny, sizeof(int));
  int* idtemp = (int*)calloc(inNum, sizeof(int));

  for(int i = 0; i < nx*ny; i++) {
    // imgMinZ[i] =  10.0f;
    // imgMaxZ[i] = -10.0f;
    imgMinZ[i] = std::numeric_limits<float>::max();
    imgMaxZ[i] = std::numeric_limits<float>::lowest();
    imgMeanZ[i] = -10.0f; // 无效占位数据初值
    // imgSumZ[i] = 0.0f; // calloc已填充0
    // imgNumZ[i] = 0;    // calloc已填充0
  }

  float inv_dx = 1.0f / dx;
  float inv_dy = 1.0f / dy;

  // ROI区域栅格地图统计
  for(int i = 0; i < points_num; i++) {   
    idtemp[i] = -1;
    int idx_pts = i*4;
    int idy_pts = i*4+1;
    int idz_pts = i*4+2;
    // =======================
    // 1. 只处理ROI范围内的点
    // X: -20 ~ 20 米
    // Y: -10 ~ 10 米
    // =======================
    if ( (points_in[idx_pts] > -x_len) 
      && (points_in[idx_pts] <  x_len)
      && (points_in[idy_pts] > -y_len)
      && (points_in[idy_pts] <  y_len) ) {
      // =======================
      // 2. 计算这个点落在对应栅格
      // idx = 栅格X编号
      // idy = 栅格Y编号
      // =======================
      int idx = static_cast<int>((inPoints[idx_pts] - offx) * inv_dx);
      int idy = static_cast<int>((inPoints[idy_pts] - offy) * inv_dy);
      int grid_id = idx + idy * nx;  // 把二维栅格坐标 → 转成一维下标  
      // 安全判断：越界直接跳过
      if (grid_id < 0 || grid_id >= nx*ny) {
        continue;
      }
      idtemp[i] = grid_id;  
      // =======================
      // 3. 开始统计这个栅格！
      // =======================
      imgSumZ[grid_id] += inPoints[idz_pts];
      imgNumZ[grid_id] += 1;  
      // 更新栅格最小Z
      if(inPoints[idz_pts] < imgMinZ[grid_id]) {
          imgMinZ[grid_id] = inPoints[idz_pts];
      }  
      // 更新栅格最大Z
      if(inPoints[idz_pts] > imgMaxZ[grid_id]) {
          imgMaxZ[grid_id] = inPoints[idz_pts];
      }
    }
  }

  for(int i = 0; i < points_num; i++) {
    if (outNum >= 60000) {
      break;
    }

    if (idtemp[i] > 0 && idtemp[i] < nx*ny) {
      // 计算均值高度
      imgMeanZ[idtemp[i]] = static_cast<float>(imgSumZ[idtemp[i]] / (imgNumZ[idtemp[i]] + 0.0001));
        
      // 最高点与均值高度差小于阈值0.4；点数大于3；均值高度小于1 
      if ((imgMaxZ[idtemp[i]] - imgMeanZ[idtemp[i]]) < THR 
        && imgNumZ[idtemp[i]] > 3 
        && imgMeanZ[idtemp[i]] < 2.0) {
       // imgMeanZ[idtemp[i]]<0&&
       points_out[outNum*4]   = points_in[i*4];
       points_out[outNum*4+1] = points_in[i*4+1];
       points_out[outNum*4+2] = points_in[i*4+2];
       points_out[outNum*4+3] = points_in[i*4+3];
       
       outNum++;
      }
    }
  }

  free(imgMinZ);
  free(imgMaxZ);
  free(imgSumZ);
  free(imgMeanZ);
  free(imgNumZ);
  free(idtemp);

  return outNum;
}

int PointsCorrect::calculate_ground_pos(float* ground_pos_out, float* points_in, int points_num, float search_radius)
{
  int valid_ground_patch_count = 0;

  // Enough valid points
  if(points_num > 3) {
    // Convert point cloud to PCL format
    PointXYZCloudPtr cloud_ptr(new PointXYZCloud);
    cloud_ptr->width = points_num;
    cloud_ptr->height = 1;
    cloud_ptr->points.resize(cloud_ptr->width*cloud_ptr->height);
    int offset_index = 0;
    for (int i = 0; i < cloud_ptr->points.size(); i++) {
      offset_index = i * 4;
      cloud_ptr->points[i].x = points_in[offset_index];
      cloud_ptr->points[i].y = points_in[offset_index+1];
      cloud_ptr->points[i].z = points_in[offset_index+2];
    }
  
    // Set Kdtree data with point cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud_ptr);
  
    unsigned char* point_occupied_flag_buffer = (unsigned char*)calloc(points_num, sizeof(unsigned char));
    for(int i = 0; i < points_num; i++)
    {
      if((valid_ground_patch_count < MAX_GROUND_PATCH_LIMIT) && (point_occupied_flag_buffer[i] == 0)) {
        NeighborPlanePCA npp;
        PointXYZ search_point;
        search_point.x = cloud_ptr->points[i].x;
        search_point.y = cloud_ptr->points[i].y;
        search_point.z = cloud_ptr->points[i].z;

        // Run neighborhood PCA for local plane feature
        if(PointCloudUtils::calculate_neighbor_plane_pca(npp, cloud_ptr, kdtree, search_point, search_radius) > 0) {
          // Mark all neighbor points as occupied to avoid repeated calculation
          for(int j = 0; j < static_cast<int>(npp.neighbor_indices.size()); j++) {
            point_occupied_flag_buffer[npp.neighbor_indices[j]] = 1;
          }

          // Judge whether local patch is flat ground via eigen value ratio
          // The larger the ratio, the closer it is to being flat
          float eigen_ratio = npp.eigen_values[1] / (npp.eigen_values[0] + DIVISION_EPSILON);
          if(eigen_ratio > GROUND_PLANE_EIGEN_RATIO_THR) {
            if(npp.eigen_vectors(2,0) > 0) {  // Vertically upward
              ground_pos_out[0] += npp.eigen_vectors(0, 0);
              ground_pos_out[1] += npp.eigen_vectors(1, 0);
              ground_pos_out[2] += npp.eigen_vectors(2, 0);
            }
            else {  // Vertically downward
              ground_pos_out[0] -= npp.eigen_vectors(0, 0);
              ground_pos_out[1] -= npp.eigen_vectors(1, 0);
              ground_pos_out[2] -= npp.eigen_vectors(2, 0);
            }
            ground_pos_out[3] += search_point.x;
            ground_pos_out[4] += search_point.y;
            ground_pos_out[5] += search_point.z;
            valid_ground_patch_count++;
          }
        }
      }
    }
    
    free(point_occupied_flag_buffer);

    /*
    if(valid_ground_patch_count > 0) {
      for(int i = 0; i < 6; i++) {
        ground_pos_out[i] /= static_cast<float>(valid_ground_patch_count); // 平均法向量 & 地面点云的中心
      }
      if(abs(ground_pos_out[0]) < 0.1) {
        ground_pos_out[0] = ground_pos_out[0] * (1 - abs(ground_pos_out[0])) * 4.5;
      }
      else if(abs(ground_pos_out[0]) < 0.2) {
        ground_pos_out[0] = ground_pos_out[0] * (1 - abs(ground_pos_out[0])) * 3.2;
      }
      else {
        ground_pos_out[0] = ground_pos_out[0] * (1 - abs(ground_pos_out[0])) * 2.8;
      }
      ground_pos_out[1] = ground_pos_out[1] * 2.3; 
    }
    */

    if (valid_ground_patch_count > 0) {
      // Average accumulated ground normal vector and patch center coordinates
      for (int i = 0; i < 6; i++) {
        ground_pos_out[i] /= static_cast<float>(valid_ground_patch_count);
      }

      // Non-linear adaptive correction for X component of ground normal
      // Purpose: Compensate LiDAR roll installation bias, amplify weak slope on flat ground
      // Formula: nx = nx * (1 - abs(nx)) * scale, suppress correction on steep slope
      float abs_nx = std::fabs(ground_pos_out[0]);
      if (abs_nx < NORM_X_CORR_THR1) {
        ground_pos_out[0] *= (1.0f - abs_nx) * NORM_X_SCALE_LIGHT;
      }
      else if (abs_nx < NORM_X_CORR_THR2) {
        ground_pos_out[0] *= (1.0f - abs_nx) * NORM_X_SCALE_MID;
      }
      else {
        ground_pos_out[0] *= (1.0f - abs_nx) * NORM_X_SCALE_HEAVY;
      }

      // Fixed scale compensation for Y normal component, offset fixed pitch mounting error
      ground_pos_out[1] *= NORM_Y_FIX_SCALE;
    }
  }
  else {
    // for(int i = 0; i<6; i++) {
    //   ground_pos_out[i] = 0;
    // }
    memset(ground_pos_out, 0, 6*sizeof(float));
  }

  return valid_ground_patch_count;
}

int PointsCorrect::correct_point_cloud(float* points, int points_num, float* ground_pos)
{
  // Null pointer safety guard
  if (points == nullptr || ground_pos == nullptr)
  {
    return -1;
  }

  float rtm[9] = {0.0f};
  float z_norm[3] = {0.0f,0.0f,1.0f};
  float tmp[3] = {0.0f};
  float ground_z_offset = 0.0f;
  int offset_idx = 0;

  PointCloudUtils::calculate_rotation_transformation_matrix(rtm, ground_pos, z_norm); 

  ground_z_offset = rtm[2] * ground_pos[3] + rtm[5] * ground_pos[4] + rtm[8] * ground_pos[5];

  for(int i = 0; i < points_num; i++) {
    offset_idx = i * 4;
    tmp[0] = rtm[0] * points[offset_idx] + rtm[3] * points[offset_idx+1] + rtm[6] * points[offset_idx+2];
    tmp[1] = rtm[1] * points[offset_idx] + rtm[4] * points[offset_idx+1] + rtm[7] * points[offset_idx+2];
    tmp[2] = rtm[2] * points[offset_idx] + rtm[5] * points[offset_idx+1] + rtm[8] * points[offset_idx+2] - ground_z_offset;   
    points[offset_idx]   = tmp[0];
    points[offset_idx+1] = tmp[1];
    points[offset_idx+2] = tmp[2];
  }
  
  return 0;
}

} // namespace lidar_proc
