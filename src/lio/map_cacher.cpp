#include "lio/map_cacher.hpp"

namespace lio
{
MapCacher::MapCacher(MapManager* map_mgr_ptr, float filter_corner, float filter_surf)
    : map_manager_ptr_(map_mgr_ptr), thread_running_flag_(true)
{
    // Init local map storage
    pcf_corner_from_local_.reset(new PointCloudType);
    pcf_surface_from_local_.reset(new PointCloudType);
    pcf_none_from_local_.reset(new PointCloudType);

    for(size_t i = 0; i < kLocalMapWindowSize; i++)
    {
        local_cloud_corner_map_[i].reset(new PointCloudType);
        local_cloud_surface_map_[i].reset(new PointCloudType);
        local_cloud_none_map_[i].reset(new PointCloudType);
    }

    // Voxel filter config
    down_size_corner_filter_.setLeafSize(filter_corner, filter_corner, filter_corner);
    down_size_surface_filter_.setLeafSize(filter_surf, filter_surf, filter_surf);
    down_size_none_filter_.setLeafSize(0.4, 0.4, 0.4);

    // Start flush thread
    flush_thread_ = std::thread(&MapCacher::threadMapIncrementFlush, this);
}

MapCacher::~MapCacher()
{
    thread_running_flag_ = false;
    if(flush_thread_.joinable())
    {
        flush_thread_.join();
    }
}

void MapCacher::pushDataForMap(const PointCloudTypePtr& corner,
                               const PointCloudTypePtr& surf,
                               const PointCloudTypePtr& nonfeature,
                               const Eigen::Matrix4d& transform)
{
    std::lock_guard<std::mutex> lock(queue_mtx_);
    MapUpdateTask task;
    *task.corner_cloud = *corner;
    *task.surf_cloud = *surf;
    *task.nonfeature_cloud = *nonfeature;
    task.transform = transform;
    task_queue_.push(std::move(task));
}

void MapCacher::update_local_map_increment(const PointCloudTypePtr& corner_in,
                                           const PointCloudTypePtr& surf_in,
                                           const PointCloudTypePtr& non_in,
                                           const Eigen::Matrix4d& transform)
{
  size_t corner_size = corner_in->points.size();
  size_t surf_size = surf_in->points.size();
  size_t none_size = non_in->points.size();
  PointType point_sel;  
  size_t win_id = local_map_id_ % kLocalMapWindowSize;
  local_cloud_corner_map_[win_id]->clear();
  local_cloud_surface_map_[win_id]->clear();
  local_cloud_none_map_[win_id]->clear();  
  // Transform points to world map coordinate
  for(size_t i = 0; i < corner_size; i++)
  {
      MapManager::point_associate_to_map(&corner_in->points[i], &point_sel, transform);
      local_cloud_corner_map_[win_id]->push_back(point_sel);
  }
  for(size_t i = 0; i < surf_size; i++)
  {
      MapManager::point_associate_to_map(&surf_in->points[i], &point_sel, transform);
      local_cloud_surface_map_[win_id]->push_back(point_sel);
  }
  for(size_t i = 0; i < none_size; i++)
  {
      MapManager::point_associate_to_map(&non_in->points[i], &point_sel, transform);
      local_cloud_none_map_[win_id]->push_back(point_sel);
  }  
  // Merge all local window frames
  pcf_corner_from_local_->clear();
  pcf_surface_from_local_->clear();
  pcf_none_from_local_->clear();
  for(size_t i = 0; i < kLocalMapWindowSize; i++)
  {
      *pcf_corner_from_local_ += *local_cloud_corner_map_[i];
      *pcf_surface_from_local_ += *local_cloud_surface_map_[i];
      *pcf_none_from_local_ += *local_cloud_none_map_[i];
  }  
  // Downsample local map
  PointCloudTypePtr temp(new PointCloudType());
  down_size_corner_filter_.setInputCloud(pcf_corner_from_local_);
  down_size_corner_filter_.filter(*temp);
  pcf_corner_from_local_ = temp;  
  temp->clear();
  down_size_surface_filter_.setInputCloud(pcf_surface_from_local_);
  down_size_surface_filter_.filter(*temp);
  pcf_surface_from_local_ = temp;  
  temp->clear();
  down_size_none_filter_.setInputCloud(pcf_none_from_local_);
  down_size_none_filter_.filter(*temp);
  pcf_none_from_local_ = temp;  
  local_map_id_++;
}

void MapCacher::threadMapIncrementFlush()
{
    PointCloudTypePtr tmp_cor(new PointCloudType);
    PointCloudTypePtr tmp_surf(new PointCloudType);
    PointCloudTypePtr tmp_none(new PointCloudType);
    PointCloudTypePtr batch_cor(new PointCloudType);
    PointCloudTypePtr batch_surf(new PointCloudType);
    PointCloudTypePtr batch_none(new PointCloudType);
    Eigen::Matrix4d batch_transform;

    while(thread_running_flag_)
    {
        std::unique_lock<std::mutex> locker(queue_mtx_);
        if(!task_queue_.empty())
        {
            MapUpdateTask task = task_queue_.front();
            task_queue_.pop();
            locker.unlock();

            map_update_id_++;
            // Transform feature to map frame
            map_manager_ptr_->feature_associate_to_map(
                task.corner_cloud, task.surf_cloud, task.nonfeature_cloud,
                tmp_cor, tmp_surf, tmp_none, task.transform
            );

            // Accumulate batch
            *batch_cor += *tmp_cor;
            *batch_surf += *tmp_surf;
            *batch_none += *tmp_none;
            batch_transform = task.transform;

            tmp_cor->clear();
            tmp_surf->clear();
            tmp_none->clear();

            // Flush batch to real map every N frames
            if(map_update_id_ % kMapSkipFrameCount == 0)
            {
                map_manager_ptr_->map_increment_update(batch_cor, batch_surf, batch_none, batch_transform);
                batch_cor->clear();
                batch_surf->clear();
                batch_none->clear();
            }
        }
        else
        {
            locker.unlock();
        }
        std::this_thread::sleep_for(kThreadSleepMs);
    }
}

void MapCacher::update_global_map_duplicate()
{
  std::unique_lock<std::mutex> locker(map_manager_ptr_->mtx_MapManager);

  for(size_t i = 0; i < kValidVoxelGridCount; i++) {
    global_kdtree_corner_map_[i] = map_manager_ptr_->get_kdtree_corner_map(i);
    global_kdtree_surface_map_[i] = map_manager_ptr_->get_kdtree_surface_map(i);
    global_kdtree_none_map_[i] = map_manager_ptr_->get_kdtree_none_map(i);

    global_cloud_corner_map_[i] = map_manager_ptr_->get_corner_cloud_for_match(i);
    global_cloud_surface_map_[i] = map_manager_ptr_->get_surface_cloud_for_match(i);
    global_cloud_none_map_[i] = map_manager_ptr_->get_none_cloud_for_match(i);
  }
  
  cloud_center_width_last_ = map_manager_ptr_->get_laserCloudCenWidth_last();
  cloud_center_height_last_ = map_manager_ptr_->get_laserCloudCenHeight_last();
  cloud_center_depth_last_ = map_manager_ptr_->get_laserCloudCenDepth_last();
  
  locker.unlock();
}

} // end of namespace lio