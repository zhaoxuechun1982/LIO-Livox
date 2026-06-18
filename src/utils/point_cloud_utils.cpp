#include "point_cloud_utils.hpp"
#include <Eigen/Dense>

namespace lidar_proc
{

int PointCloudUtils::calculate_neighbor_plane_pca(NeighborPlanePCA& out_npp,
                                                  PointXYZCloudPtr cloud_ptr,
                                                  KdTreePtr kdtree,
                                                  PointXYZ search_point,
                                                  float search_radius)
{
  std::vector<float> dist_sq;
  // PointXYZCloudPtr sub_cloud_ptr(new PointXYZCloud);

  if (kdtree.radiusSearch(search_point, search_radius, out_npp.neighbor_indices, dist_sq) > PCA_MIN_NEIGHBOR_COUNT) {
    Eigen::Vector4f centroid;
	Eigen::Matrix3f covariance;

    // 构建搜索邻域内的点云
    // sub_cloud_ptr->width = out_npp.neighbor_indices.size();
    // sub_cloud_ptr->height = 1;
    // sub_cloud_ptr->points.resize(sub_cloud_ptr->width*sub_cloud_ptr->height);
    // pcl::copyPointCloud(*cloud_ptr, out_npp.neighbor_indices, *sub_cloud_ptr);
    // for (int i = 0; i < sub_cloud_ptr->points.size(); i++) {
    //   sub_cloud_ptr->points[i].x = cloud_ptr->points[out_npp.neighbor_indices[i]].x;
    //   sub_cloud_ptr->points[i].y = cloud_ptr->points[out_npp.neighbor_indices[i]].y;
    //   sub_cloud_ptr->points[i].z = cloud_ptr->points[out_npp.neighbor_indices[i]].z;
    // }

    // 利用PCA主元分析法获得点云的三个主方向，计算质心、协方差矩阵
    //pcl::compute3DCentroid(*sub_cloud_ptr, centroid);
	//pcl::computeCovarianceMatrixNormalized(*sub_cloud_ptr, centroid, covariance);
    pcl::compute3DCentroid(*cloud_ptr, out_npp.neighbor_indices, centroid);
    pcl::computeCovarianceMatrixNormalized(*cloud_ptr, out_npp.neighbor_indices, centroid, covariance);

    // 计算协方差矩阵的特征值和特征向量，最小特征向量即为平面法向
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    if (eigen_solver.info() != Eigen::Success) {
      // to do...打印信息
      // 协方差矩阵奇异，求解失败，清空输出防止脏数据
      out_npp.neighbor_indices.clear();
      out_npp.eigen_vectors.setZero();
      out_npp.eigen_values.setZero();
    }
    else {
	  out_npp.eigen_vectors = eigen_solver.eigenvectors();
	  out_npp.eigen_values = eigen_solver.eigenvalues();
      float denominator = out_npp.eigen_values.sum() + DIVISION_EPSILON;
      out_npp.eigen_values(0) /= denominator;
      out_npp.eigen_values(1) /= denominator;
      out_npp.eigen_values(2) /= denominator;
    }
  }
  else {
    // 邻域点数量不足，无有效平面特征
    out_npp.neighbor_indices.clear();
    out_npp.eigen_vectors.setZero();
    out_npp.eigen_values.setZero();
  }

  return out_npp.neighbor_indices.size();
}

int PointCloudUtils::calculate_rotation_transformation_matrix(float* rtm, float* v0, float* v1)
{
  // Normalize v0
  float nv0 = std::sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
  v0[0] /= (nv0 + DIVISION_EPSILON);
  v0[1] /= (nv0 + DIVISION_EPSILON);
  v0[2] /= (nv0 + DIVISION_EPSILON);

  // Normalize v1
  float nv1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
  v1[0] /= (nv1 + DIVISION_EPSILON);
  v1[1] /= (nv1 + DIVISION_EPSILON);
  v1[2] /= (nv1 + DIVISION_EPSILON);

  // Cross product to get rotation axis v2
  float v2[3];
  v2[0] = v0[1] * v1[2] - v0[2] * v1[1];
  v2[1] = v0[2] * v1[0] - v0[0] * v1[2];
  v2[2] = v0[0] * v1[1] - v0[1] * v1[0];

  // Cosine value and sine value
  float ca = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
  float sa = std::sqrt(1.0f - ca * ca);

  // Rodrigues formula fill matrix
  rtm[0] = v2[0] * v2[0] * (1.0f - ca) + ca;
  rtm[4] = v2[1] * v2[1] * (1.0f - ca) + ca;
  rtm[8] = v2[2] * v2[2] * (1.0f - ca) + ca;

  rtm[1] = rtm[3] = v2[0] * v2[1] * (1.0f - ca);
  rtm[2] = rtm[6] = v2[0] * v2[2] * (1.0f - ca);
  rtm[5] = rtm[7] = v2[1] * v2[2] * (1.0f - ca);

  rtm[1] +=  v2[2] * sa;
  rtm[2] += -v2[1] * sa;
  rtm[3] += -v2[2] * sa;

  rtm[5] +=  v2[0] * sa;
  rtm[6] +=  v2[1] * sa;
  rtm[7] += -v2[0] * sa;

  return 0;
}

} // namespace lidar_proc