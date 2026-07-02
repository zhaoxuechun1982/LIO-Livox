#ifndef LIO_MAP_CACHER_HPP
#define LIO_MAP_CACHER_HPP

#include "type/point_cloud_type.hpp"
#include "lio/map_manager.hpp"

#include <Eigen/Core>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>

using namespace lio_data_type;

namespace lio
{
struct MapUpdateTask
{
  PointCloudTypePtr corner_cloud;
  PointCloudTypePtr surf_cloud;
  PointCloudTypePtr nonfeature_cloud;
  Eigen::Matrix4d transform;  
  MapUpdateTask()
  {
    corner_cloud.reset(new PointCloudType);
    surf_cloud.reset(new PointCloudType);
    nonfeature_cloud.reset(new PointCloudType);
    transform.setIdentity();
  }
};

class MapCacher
{
public:
    static constexpr double kValidPlaneTangentWeight = 0.0003;
    static constexpr double kNoValidPlaneTangentWeight = 0.0;
    static constexpr double kShortSearchDistThreshold = 1.0;
    static constexpr double kMedianSearchDistThreshold = 10.0;
    static constexpr double kLongSearchDistThreshold = 25.0;   
    static constexpr unsigned int kMapSkipFrameCount = 2;
    static constexpr std::chrono::milliseconds kThreadSleepMs{2};

public:
    MapCacher(MapManager* map_mgr_ptr, float filter_corner, float filter_surf);
    ~MapCacher();

    // Push frame feature data to cache queue (forMap: buffer only)
    void pushDataForMap(const PointCloudTypePtr& corner,
                        const PointCloudTypePtr& surf,
                        const PointCloudTypePtr& nonfeature,
                        const Eigen::Matrix4d& transform);

    // Local map incremental update logic, moved from PoseEstimator
    void update_local_map_Increment(const PointCloudTypePtr& corner_in,
                                    const PointCloudTypePtr& surf_in,
                                    const PointCloudTypePtr& non_in,
                                    const Eigen::Matrix4d& transform);

    // Read local cached map for matching (used by PoseEstimator)
    inline PointCloudTypePtr getLocalCornerMap()  { return pcf_corner_from_local_; }
    inline PointCloudTypePtr getLocalSurfMap()     { return pcf_surface_from_local_; }
    inline PointCloudTypePtr getLocalNonFeatureMap(){ return pcf_none_from_local_; }

    void update_global_map_duplicate();
    inline void set_param_plane_tangent_weight(double value) { param_plane_tangent_weight_ = value; }
    inline void set_param_search_dist_threshold(double value) { param_search_dist_threshold_ = value; }
    //inline const double& get_param_plane_tangent_weight() const { return param_plane_tangent_weight_; }
    //inline const int& get_cloud_corner_width_last() const { return cloud_center_width_last_; }
    //inline const int& get_cloud_corner_height_last() const { return cloud_center_height_last_; }
    //inline const int& get_cloud_corner_depth_last() const { return cloud_center_depth_last_; }
    //inline const PointKdTreeType& get_global_kdtree_corner_map(int i) const { return global_kdtree_corner_map_[i];} 
    //inline const PointKdTreeType& get_global_kdtree_surface_map(int i) const { return global_kdtree_surface_map_[i];} 
    //inline const PointKdTreeType& get_global_kdtree_none_map(int i) const { return global_kdtree_none_map_[i];} 
    //inline const PointCloudType& get_global_cloud_corner_map(int i) const { return global_cloud_corner_map_[i];} 
    //inline const PointCloudType& get_global_cloud_surface_map(int i) const { return global_cloud_surface_map_[i];} 
    //inline const PointCloudType& get_global_cloud_none_map(int i) const { return global_cloud_none_map_[i];} 

private:
  // Async thread for map flush to MapManager (toMap: real write)
  void threadMapIncrementFlush();


  MapManager* map_manager_ptr_;
  // Cache task queue & sync
  std::queue<MapUpdateTask> task_queue_;
  std::mutex queue_mtx_;
  std::thread flush_thread_;
  bool thread_running_flag_;
  unsigned int map_update_id_ = 0;

  // Local sliding map buffer (moved from PoseEstimator)
  static constexpr size_t kLocalMapWindowSize = 50;
  size_t local_map_id_ = 0;
  PointCloudTypePtr local_cloud_corner_map_[kLocalMapWindowSize];
  PointCloudTypePtr local_cloud_surface_map_[kLocalMapWindowSize];
  PointCloudTypePtr local_cloud_none_map_[kLocalMapWindowSize]; 
  PointCloudTypePtr pcf_corner_from_local_;
  PointCloudTypePtr pcf_surface_from_local_;
  PointCloudTypePtr pcf_none_from_local_;
  // Downsample filters for local map
  PointVoxelGridType down_size_corner_filter_;
  PointVoxelGridType down_size_surface_filter_;
  PointVoxelGridType down_size_none_filter_;

  friend class LidarFeatureMatcher;
  static constexpr size_t kValidVoxelGridCount = 4851;
  static constexpr size_t kGlobalMapWindowSize = 10000;
  PointKdTreeType global_kdtree_corner_map_[kGlobalMapWindowSize];
  PointKdTreeType global_kdtree_surface_map_[kGlobalMapWindowSize];
  PointKdTreeType global_kdtree_none_map_[kGlobalMapWindowSize];
  PointCloudType global_cloud_corner_map_[kGlobalMapWindowSize];
  PointCloudType global_cloud_surface_map_[kGlobalMapWindowSize];
  PointCloudType global_cloud_none_map_[kGlobalMapWindowSize];
  int cloud_center_width_last_ = 10;
  int cloud_center_height_last_ = 5;
  int cloud_center_depth_last_ = 10;


  double param_plane_tangent_weight_ = 0.0;
  double param_search_dist_threshold_ = 1.0;
};

} // end of namespace lio
#endif // end of LIO_MAP_CACHER_HPP