#ifndef LIO_MAP_MANAGER_H
#define LIO_MAP_MANAGER_H

#include "type/point_cloud_type.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <future>

using namespace lio_data_type;
namespace lio 
{

class MapManager
{
public:

    std::mutex mtx_MapManager;
    /** \brief constructor of MapManager */
    MapManager(const float& filter_corner, const float& filter_surf);

    static size_t ToIndex(int i, int j, int k);

    /** \brief transform float to int
  */
    static uint32_t _float_as_int(float f)
    {
      union{uint32_t i; float f;} conv{};
      conv.f = f;
      return conv.i;
    }

    /** \brief transform int to float
      */
    static float _int_as_float(uint32_t i)
    {
      union{float f; uint32_t i;} conv{};
      conv.i = i;
      return conv.f;
    }

    /** \brief transform point pi to the MAP coordinate
     * \param[in] pi: point to be transformed
     * \param[in] po: point after transfomation
     * \param[in] _transformTobeMapped: transform matrix between pi and po
     */
    static void point_associate_to_map(PointType const * const pi,
                                    PointType * const po,
                                    const Eigen::Matrix4d& _transformTobeMapped);

    void feature_associate_to_map(const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudCornerToMap,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudSurfToMap,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureToMap,
                               const Eigen::Matrix4d& transformTobeMapped);
    /** \brief add new lidar points to the map
     * \param[in] laserCloudCornerStack: coner feature points that need to be added to map
     * \param[in] laserCloudSurfStack: surf feature points that need to be added to map
     * \param[in] transformTobeMapped: transform matrix of the lidar pose
     */
    void map_increment_update(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
                      const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
                      const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
                      const Eigen::Matrix4d& transformTobeMapped);

    /** \brief retrieve map points according to the lidar pose
     * \param[in] laserCloudCornerFromMap: store coner feature points retrieved from map
     * \param[in] laserCloudSurfFromMap: tore surf feature points retrieved from map
     * \param[in] transformTobeMapped: transform matrix of the lidar pose
     */
    void MapMove(const Eigen::Matrix4d& transformTobeMapped);


    size_t FindUsedCornerMap(const PointType *p,int a,int b,int c);

    size_t FindUsedSurfMap(const PointType *p,int a,int b,int c);

    size_t FindUsedNonFeatureMap(const PointType *p,int a,int b,int c);

    PointKdTreeType get_kdtree_corner_map(int i){
      return kdtree_corner_map_last_[i];
    }
    PointKdTreeType get_kdtree_surface_map(int i){
      return kdtree_surface_map_last_[i];
    }
    PointKdTreeType get_kdtree_none_map(int i){
      return kdtree_none_map_last_[i];
    }
		pcl::PointCloud<PointType>::Ptr get_corner_map(){
			return laserCloudCornerFromMap;
		}
		pcl::PointCloud<PointType>::Ptr get_surf_map(){
			return laserCloudSurfFromMap;
		}
    pcl::PointCloud<PointType>::Ptr get_nonfeature_map(){
			return laserCloudNonFeatureFromMap;
		}
    int get_map_current_pos(){
      return currentUpdatePos;
    }
    int get_laserCloudCenWidth_last(){
      return laserCloudCenWidth_last;
    }
    int get_laserCloudCenHeight_last(){
      return laserCloudCenHeight_last;
    }
    int get_laserCloudCenDepth_last(){
      return laserCloudCenDepth_last;
    }

    inline const PointCloudType& get_surface_cloud_for_match(size_t i) const
    {
      return surface_cloud_for_match_[i];
    }

    inline const PointCloudType& get_corner_cloud_for_match(size_t i) const
    {
      return corner_cloud_for_match_[i];
    }

    inline const PointCloudType& get_none_cloud_for_match(size_t i) const
    {
      return none_cloud_for_match_[i];
    }

private:
    int laserCloudCenWidth = 10;
    int laserCloudCenHeight = 5;
    int laserCloudCenDepth = 10;

    int laserCloudCenWidth_last = 10;
    int laserCloudCenHeight_last = 5;
    int laserCloudCenDepth_last = 10;

    static const int laserCloudWidth = 21;
    static const int laserCloudHeight = 11;
    static const int laserCloudDepth = 21;
    static const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;//4851
    pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudCornerArrayStack[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudSurfArrayStack[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureArrayStack[laserCloudNum];

    static constexpr size_t kVlidVoxelGridCount = 4581;
    PointCloudType surface_cloud_for_match_[kVlidVoxelGridCount];
    PointCloudType corner_cloud_for_match_[kVlidVoxelGridCount];
    PointCloudType none_cloud_for_match_[kVlidVoxelGridCount];

    PointVoxelGridType downSizeFilterCorner;
    PointVoxelGridType downSizeFilterSurf;
    PointVoxelGridType downSizeFilterNonFeature;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromMap;

    PointKdTreeType::Ptr laserCloudCornerKdMap[laserCloudNum];
    PointKdTreeType::Ptr laserCloudSurfKdMap[laserCloudNum];
    PointKdTreeType::Ptr laserCloudNonFeatureKdMap[laserCloudNum];

    PointKdTreeType CornerKdMap_copy[laserCloudNum];
    PointKdTreeType SurfKdMap_copy[laserCloudNum];
    PointKdTreeType NonFeatureKdMap_copy[laserCloudNum];

    PointKdTreeType kdtree_corner_map_last_[laserCloudNum];
    PointKdTreeType kdtree_surface_map_last_[laserCloudNum];
    PointKdTreeType kdtree_none_map_last_[laserCloudNum];

    static constexpr size_t kLocalMapWindowSize = 60;
    pcl::PointCloud<PointType>::Ptr local_corner_map_[kLocalMapWindowSize];
    pcl::PointCloud<PointType>::Ptr local_surface_map_[kLocalMapWindowSize];
    pcl::PointCloud<PointType>::Ptr local_none_map_[kLocalMapWindowSize];

    int localMapID = 0;
    int currentUpdatePos = 0;
    int estimatorPos = 0;
};

}



#endif //LIO_MAP_MANAGER_H
