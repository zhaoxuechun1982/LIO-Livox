#ifndef LIO_LIDAR_FEATURE_TYPE_HPP
#define LIO_LIDAR_FEATURE_TYPE_HPP

#include <vector>
#include <Eigen/Core>

namespace lio_data_type
{
  using FeatureLineVector = std::vector<FeatureLine>;
  using FeaturePlaneVector = std::vector<FeaturePlane>;
  using FeaturePlaneVectorVector = std::vector<FeaturePlaneVector>;
  using FeatureNoneVector = std::vector<FeatureNone>;

  struct FeatureLine 
  {
  	Eigen::Vector3d point_origin;
  	Eigen::Vector3d line_p1;
  	Eigen::Vector3d line_p2;
  	double error;
  	bool valid;

    // Constructor
  	FeatureLine(const Eigen::Vector3d& point_origin_, 
                const Eigen::Vector3d& line_point1_, 
                const Eigen::Vector3d& line_point2_):
      point_origin(point_origin_), 
      line_p1(line_point1_), 
      line_p2(line_point2_),
      error(0.0),
      valid(false)
    { 
    }

  	void calculate_error(const Eigen::Matrix4d& pose)
    {
      Eigen::Vector3d p_to_map = pose.topLeftCorner(3,3) * point_origin + pose.topRightCorner(3,1);
      Eigen::Vector3d v_p1_p0 = p_to_map - line_p1;
      Eigen::Vector3d v_p1_p2 = line_p2 - line_p1;
      double norm_p1_p2 = v_p1_p2.norm();

      if (norm_p1_p2 >= 1e-6) {
        error = v_p1_p0.cross(v_p1_p2).norm() / norm_p1_p2;
      }
      else {
        error = 1e-5;
      }
  	}
  };

  struct FeaturePlane 
  {
  	Eigen::Vector3d point_origin;
  	double param_a;
  	double param_b;
  	double param_c;
  	double param_d;
  	double error;
  	bool valid;
  	FeaturePlane(const Eigen::Vector3d& point_origin, 
                 const double& param_a_, 
                 const double& param_b_, 
                 const double& param_c_, 
                 const double& param_d_):
      point_origin(point_origin), 
      param_a(param_a_), 
      param_b(param_b_), 
      param_c(param_c_), 
      param_d(param_d_),
      error(0.0),
      valid(false)
    {}

  	void calculate_error(const Eigen::Matrix4d& pose) 
    {
  		Eigen::Vector3d p_to_map = pose.topLeftCorner(3,3) * point_origin + pose.topRightCorner(3,1);
  		error = param_a * p_to_map(0) + param_b * p_to_map(1) + param_c * p_to_map(2) + param_d;
  	}
  };

  struct FeaturePlaneVector 
  {
  	Eigen::Vector3d point_origin;
  	Eigen::Vector3d point_project;
  	Eigen::Matrix3d sqrt_info;
  	double error;
  	bool valid;

  	FeaturePlaneVector(const Eigen::Vector3d& point_origin_, 
                       const Eigen::Vector3d& point_project_, 
                       const Eigen::Matrix3d& sqrt_info_) :
      point_origin(point_origin_), 
      point_project(point_project_), 
      sqrt_info(sqrt_info_),
      error(0.0),
      valid(false)
    {}

  	void calculate_error(const Eigen::Matrix4d& pose) 
    {
  		Eigen::Vector3d p_to_map = pose.topLeftCorner(3,3) * point_origin + pose.topRightCorner(3,1);
      // error = (p_to_map - point_project).norm();
  		error = (sqrt_info* (p_to_map - point_project)).norm();
  	}
  };

  struct FeatureNone 
  {
  	Eigen::Vector3d point_origin;
  	double param_a;
  	double param_b;
  	double param_c;
  	double param_d;
  	double error;
  	bool valid;

  	FeatureNone(const Eigen::Vector3d& point_origin_, 
                const double& param_a_, 
                const double& param_b_, 
                const double& param_c_, 
                const double& param_d_) :
      point_origin(point_origin_), 
      param_a(param_a_), 
      param_b(param_b_), 
      param_c(param_c_), 
      param_d(param_d_),
  		error(0.0),
      valid(false)
    {}

  	void calculate_error(const Eigen::Matrix4d& pose)
    {
  		Eigen::Vector3d p_to_map = pose.topLeftCorner(3,3) * point_origin + pose.topRightCorner(3,1);
  		error = param_a * p_to_map(0) + param_b * p_to_map(1) + param_c * p_to_map(2) + param_d;
  	}
  };
} // end of namespace lio_data_type

#endif // end of LIO_LIDAR_FEATURE_TYPE_HPP