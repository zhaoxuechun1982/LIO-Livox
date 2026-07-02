#include "lio/map_manager.hpp"
#include <fstream>

namespace lio
{
MapManager::MapManager(const float& filter_corner, const float& filter_surf) 
{
  for (int i = 0; i < laserCloudNum; i++) {
    laserCloudCornerArray[i].reset(new PointCloudType());
    laserCloudSurfArray[i].reset(new PointCloudType());
    laserCloudNonFeatureArray[i].reset(new PointCloudType());
    laserCloudCornerArrayStack[i].reset(new PointCloudType());
    laserCloudSurfArrayStack[i].reset(new PointCloudType());
    laserCloudNonFeatureArrayStack[i].reset(new PointCloudType());

    laserCloudCornerKdMap[i].reset(new PointKdTreeType);
    laserCloudSurfKdMap[i].reset(new PointKdTreeType);
    laserCloudNonFeatureKdMap[i].reset(new PointKdTreeType);
  }

  for (int i = 0; i < kLocalMapWindowSize; i++) {
    local_corner_map_[i].reset(new PointCloudType());
    local_surface_map_[i].reset(new PointCloudType());
    local_none_map_[i].reset(new PointCloudType());
  }

  laserCloudCornerFromMap.reset(new PointCloudType());
  laserCloudSurfFromMap.reset(new PointCloudType());
  laserCloudNonFeatureFromMap.reset(new PointCloudType());
  
  downSizeFilterCorner.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilterNonFeature.setLeafSize(0.4, 0.4, 0.4);
}

size_t MapManager::ToIndex(int i, int j, int k) 
{
  return i + kCloudDepth * j + kCloudDepth * kCloudWidth * k;
}

/** \brief transform point pi to the MAP coordinate
 * \param[in] pi: point to be transformed
 * \param[in] po: point after transfomation
 * \param[in] _transformTobeMapped: transform matrix between pi and po
 */
void MapManager::point_associate_to_map(PointType const * const pi,
                                        PointType * const po,
                                        const Eigen::Matrix4d& _transformTobeMapped) 
{
        Eigen::Vector3d pin, pout;
        pin.x() = pi->x;
        pin.y() = pi->y;
        pin.z() = pi->z;
        pout = _transformTobeMapped.topLeftCorner(3,3) * pin + _transformTobeMapped.topRightCorner(3,1);
        po->x = pout.x();
        po->y = pout.y();
        po->z = pout.z();
        po->intensity = pi->intensity;
        po->normal_z = pi->normal_z;
}
void MapManager::feature_associate_to_map(const PointCloudType::Ptr& laserCloudCorner,
                                          const PointCloudType::Ptr& laserCloudSurf,
                                          const PointCloudType::Ptr& laserCloudNonFeature,
                                          const PointCloudType::Ptr& laserCloudCornerToMap,
                                          const PointCloudType::Ptr& laserCloudSurfToMap,
                                          const PointCloudType::Ptr& laserCloudNonFeatureToMap,
                                          const Eigen::Matrix4d& transformTobeMapped)
{

  int laserCloudCornerNum = laserCloudCorner->points.size();
  int laserCloudSurfNum = laserCloudSurf->points.size();
  int laserCloudNonFeatureNum = laserCloudNonFeature->points.size();
  PointType pointSel1,pointSel2,pointSel3;
  for (int i = 0; i < laserCloudCornerNum; i++) {
    point_associate_to_map(&laserCloudCorner->points[i], &pointSel1, transformTobeMapped);
    laserCloudCornerToMap->push_back(pointSel1);
  }
  for (int i = 0; i < laserCloudSurfNum; i++) {
    point_associate_to_map(&laserCloudSurf->points[i], &pointSel2, transformTobeMapped);
    laserCloudSurfToMap->push_back(pointSel2);
  }
  for (int i = 0; i < laserCloudNonFeatureNum; i++) {
    point_associate_to_map(&laserCloudNonFeature->points[i], &pointSel3, transformTobeMapped);
    laserCloudNonFeatureToMap->push_back(pointSel3);
  }
}
/** \brief add new lidar points to the map
 * \param[in] laserCloudCornerStack: coner feature points that need to be added to map
 * \param[in] laserCloudSurfStack: surf feature points that need to be added to map
 * \param[in] transformTobeMapped: transform matrix of the lidar pose
 */
void MapManager::map_increment_update(const PointCloudType::Ptr& laserCloudCornerStack,
                               const PointCloudType::Ptr& laserCloudSurfStack,
                               const PointCloudType::Ptr& laserCloudNonFeatureStack,
                               const Eigen::Matrix4d& transformTobeMapped){
  
  clock_t t0,t1,t2,t3,t4,t5;
  t0 = clock();
  std::unique_lock<std::mutex> locker2(mtx_MapManager);
  for(int i = 0; i < laserCloudNum; i++){
    kdtree_corner_map_last_[i] = *laserCloudCornerKdMap[i];
    kdtree_surface_map_last_[i] = *laserCloudSurfKdMap[i];
    kdtree_none_map_last_[i] = *laserCloudNonFeatureKdMap[i];
    surface_cloud_for_match_[i] = *laserCloudSurfArray[i];
    corner_cloud_for_match_[i] = *laserCloudCornerArray[i];
    none_cloud_for_match_[i] = *laserCloudNonFeatureArray[i];
  }

  laserCloudCenWidth_last = laserCloudCenWidth;
  laserCloudCenHeight_last = laserCloudCenHeight;
  laserCloudCenDepth_last = laserCloudCenDepth;

  locker2.unlock();
  
  t1 = clock();
  MapMove(transformTobeMapped);

  t2 = clock();
  int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
  int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
  int laserCloudNonFeatureStackNum = laserCloudNonFeatureStack->points.size();
  bool CornerChangeFlag[laserCloudNum] = {false};
  bool SurfChangeFlag[laserCloudNum] = {false};
  bool NonFeatureChangeFlag[laserCloudNum] = {false};
  PointType pointSel;
  for (int i = 0; i < laserCloudCornerStackNum; i++) {

    pointSel = laserCloudCornerStack->points[i];

    int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenDepth;
    int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenWidth;
    int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenHeight;

    if (pointSel.x + 25.0 < 0) cubeI--;
    if (pointSel.y + 25.0 < 0) cubeJ--;
    if (pointSel.z + 25.0 < 0) cubeK--;

    if (cubeI >= 0 && cubeI < kCloudDepth &&
        cubeJ >= 0 && cubeJ < kCloudWidth &&
        cubeK >= 0 &&
        cubeK < kCloudHeight) {
      size_t cubeInd = ToIndex(cubeI, cubeJ, cubeK);
      laserCloudCornerArray[cubeInd]->push_back(pointSel);
      CornerChangeFlag[cubeInd] = true;
    }
  }

  for (int i = 0; i < laserCloudSurfStackNum; i++) {
    pointSel = laserCloudSurfStack->points[i];
    int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenDepth;
    int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenWidth;
    int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenHeight;

    if (pointSel.x + 25.0 < 0) cubeI--;
    if (pointSel.y + 25.0 < 0) cubeJ--;
    if (pointSel.z + 25.0 < 0) cubeK--;

    if (cubeI >= 0 && cubeI < kCloudDepth &&
        cubeJ >= 0 && cubeJ < kCloudWidth &&
        cubeK >= 0 && cubeK < kCloudHeight) {
      size_t cubeInd = ToIndex(cubeI, cubeJ, cubeK);
      laserCloudSurfArray[cubeInd]->push_back(pointSel);
      SurfChangeFlag[cubeInd] = true;
    }
  }

  for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
    pointSel = laserCloudNonFeatureStack->points[i];
    int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenDepth;
    int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenWidth;
    int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenHeight;

    if (pointSel.x + 25.0 < 0) cubeI--;
    if (pointSel.y + 25.0 < 0) cubeJ--;
    if (pointSel.z + 25.0 < 0) cubeK--;

    if (cubeI >= 0 && cubeI < kCloudDepth &&
        cubeJ >= 0 && cubeJ < kCloudWidth &&
        cubeK >= 0 && cubeK < kCloudHeight) {
      size_t cubeInd = ToIndex(cubeI, cubeJ, cubeK);
      laserCloudNonFeatureArray[cubeInd]->push_back(pointSel);
      NonFeatureChangeFlag[cubeInd] = true;
    }
  }

  t3 = clock();
  
  laserCloudCornerFromMap->clear();
  laserCloudSurfFromMap->clear();
  laserCloudNonFeatureFromMap->clear();
  for(int i = 0; i < laserCloudNum; i++){
    if(CornerChangeFlag[i]){
      if(laserCloudCornerArray[i]->points.size() > 300){
        downSizeFilterCorner.setInputCloud(laserCloudCornerArray[i]);
        laserCloudCornerArrayStack[i]->clear();
        downSizeFilterCorner.filter(*laserCloudCornerArrayStack[i]);
        PointCloudType::Ptr tmp = laserCloudCornerArrayStack[i];
        laserCloudCornerArrayStack[i] = laserCloudCornerArray[i];
        laserCloudCornerArray[i] = tmp;
      }

      laserCloudCornerKdMap[i]->setInputCloud(laserCloudCornerArray[i]); 
      *laserCloudCornerFromMap += *laserCloudCornerKdMap[i]->getInputCloud();
    }

    if(SurfChangeFlag[i]){
      if(laserCloudSurfArray[i]->points.size() > 300){
        downSizeFilterSurf.setInputCloud(laserCloudSurfArray[i]);
        laserCloudSurfArrayStack[i]->clear();
        downSizeFilterSurf.filter(*laserCloudSurfArrayStack[i]);
        PointCloudType::Ptr tmp = laserCloudSurfArrayStack[i];
        laserCloudSurfArrayStack[i] = laserCloudSurfArray[i];
        laserCloudSurfArray[i] = tmp;
      }

      laserCloudSurfKdMap[i]->setInputCloud(laserCloudSurfArray[i]);
      *laserCloudSurfFromMap += *laserCloudSurfKdMap[i]->getInputCloud();
    }

    if(NonFeatureChangeFlag[i]){
      if(laserCloudNonFeatureArray[i]->points.size() > 300){
        downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureArray[i]);
        laserCloudNonFeatureArrayStack[i]->clear();
        downSizeFilterNonFeature.filter(*laserCloudNonFeatureArrayStack[i]);
        PointCloudType::Ptr tmp = laserCloudNonFeatureArrayStack[i];
        laserCloudNonFeatureArrayStack[i] = laserCloudNonFeatureArray[i];
        laserCloudNonFeatureArray[i] = tmp;
      }

      laserCloudNonFeatureKdMap[i]->setInputCloud(laserCloudNonFeatureArray[i]);
      *laserCloudNonFeatureFromMap += *laserCloudNonFeatureKdMap[i]->getInputCloud();
    }
      
  }

  t4 = clock();
  std::unique_lock<std::mutex> locker(mtx_MapManager);
  for(int i = 0; i < laserCloudNum; i++){
    CornerKdMap_copy[i] = *laserCloudCornerKdMap[i];
    SurfKdMap_copy[i] = *laserCloudSurfKdMap[i];
    NonFeatureKdMap_copy[i] = *laserCloudNonFeatureKdMap[i];
  }

  locker.unlock();
  t5 = clock();

  currentUpdatePos ++;

}

/** \brief move the map index if need
 * \param[in] transformTobeMapped: transform matrix of the lidar pose
 */
void MapManager::MapMove(const Eigen::Matrix4d& transformTobeMapped){
  const Eigen::Matrix3d transformTobeMapped_R = transformTobeMapped.topLeftCorner(3, 3);
  const Eigen::Vector3d transformTobeMapped_t = transformTobeMapped.topRightCorner(3, 1);

  PointType pointOnYAxis;
  pointOnYAxis.x = 0.0;
  pointOnYAxis.y = 0.0;
  pointOnYAxis.z = 10.0;

  point_associate_to_map(&pointOnYAxis, &pointOnYAxis, transformTobeMapped);

  int centerCubeI = int((transformTobeMapped_t.x() + 25.0) / 50.0) + laserCloudCenDepth;
  int centerCubeJ = int((transformTobeMapped_t.y() + 25.0) / 50.0) + laserCloudCenWidth;
  int centerCubeK = int((transformTobeMapped_t.z() + 25.0) / 50.0) + laserCloudCenHeight;

  if (transformTobeMapped_t.x() + 25.0 < 0) centerCubeI--;
  if (transformTobeMapped_t.y() + 25.0 < 0) centerCubeJ--;
  if (transformTobeMapped_t.z() + 25.0 < 0) centerCubeK--;

  while (centerCubeI < 8) {
    for (int j = 0; j < kCloudWidth; j++) {
      for (int k = 0; k < kCloudHeight; k++) {
        int i = kCloudDepth - 1;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];

        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];

        for (; i >= 1; i--) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i - 1, j, k);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        //此时i已经移动至0
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeI++;
    laserCloudCenDepth++;
  }

  while (centerCubeI >= kCloudDepth - 8) {
    for (int j = 0; j < kCloudWidth; j++) {
      for (int k = 0; k < kCloudHeight; k++) {
        int i = 0;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];
        
        for (; i < kCloudDepth - 1; i++) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i + 1, j, k);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeI--;
    laserCloudCenDepth--;
  }

  while (centerCubeJ < 8) {
    for (int i = 0; i < kCloudDepth; i++) {
      for (int k = 0; k < kCloudHeight; k++) {
        int j = kCloudWidth - 1;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];
        for (; j >= 1; j--) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i, j - 1, k);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeJ++;
    laserCloudCenWidth++;
  }

  while (centerCubeJ >= kCloudWidth - 8) {
    for (int i = 0; i < kCloudDepth; i++) {
      for (int k = 0; k < kCloudHeight; k++) {
        int j = 0;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];
        for (; j < kCloudWidth - 1; j++) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i, j + 1, k);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeJ--;
    laserCloudCenWidth--;
  }

  while (centerCubeK < 8) {
    for (int i = 0; i < kCloudDepth; i++) {
      for (int j = 0; j < kCloudWidth; j++) {
        int k = kCloudHeight - 1;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];
        for (; k >= 1; k--) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i, j, k - 1);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeK++;
    laserCloudCenHeight++;
  }

  while (centerCubeK >= kCloudHeight - 8) {
    for (int i = 0; i < kCloudDepth; i++) {
      for (int j = 0; j < kCloudWidth; j++) {
        int k = 0;
        PointKdTreeType::Ptr laserCloudCubeCornerPointerKd =
                laserCloudCornerKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeSurfPointerKd =
                laserCloudSurfKdMap[ToIndex(i, j, k)];
        PointKdTreeType::Ptr laserCloudCubeNonFeaturePointerKd =
                laserCloudNonFeatureKdMap[ToIndex(i, j, k)];

        PointCloudType::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[ToIndex(i, j, k)];
        PointCloudType::Ptr laserCloudCubeNonFeaturePointer =
                laserCloudNonFeatureArray[ToIndex(i, j, k)];
        for (; k < kCloudHeight - 1; k++) {
          const size_t index_a = ToIndex(i, j, k);
          const size_t index_b = ToIndex(i, j, k + 1);
          laserCloudCornerKdMap[index_a] = laserCloudCornerKdMap[index_b];
          laserCloudSurfKdMap[index_a] = laserCloudSurfKdMap[index_b];
          laserCloudNonFeatureKdMap[index_a] = laserCloudNonFeatureKdMap[index_b];

          laserCloudCornerArray[index_a] = laserCloudCornerArray[index_b];
          laserCloudSurfArray[index_a] = laserCloudSurfArray[index_b];
          laserCloudNonFeatureArray[index_a] = laserCloudNonFeatureArray[index_b];
        }
        laserCloudCornerKdMap[ToIndex(i, j, k)] = laserCloudCubeCornerPointerKd;
        laserCloudSurfKdMap[ToIndex(i, j, k)] = laserCloudCubeSurfPointerKd;
        laserCloudNonFeatureKdMap[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointerKd;

        laserCloudCornerArray[ToIndex(i, j, k)] = laserCloudCubeCornerPointer;
        laserCloudSurfArray[ToIndex(i, j, k)] = laserCloudCubeSurfPointer;
        laserCloudNonFeatureArray[ToIndex(i, j, k)] = laserCloudCubeNonFeaturePointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
        laserCloudCubeNonFeaturePointer->clear();
      }
    }

    centerCubeK--;
    laserCloudCenHeight--;
  }

}

size_t MapManager::find_map_used(const PointType *p, int a, int b, int c)
{
  int cubeI = int((p->x + 25.0) / 50.0) + c;
  int cubeJ = int((p->y + 25.0) / 50.0) + a;
  int cubeK = int((p->z + 25.0) / 50.0) + b;

  size_t cubeInd = 0;

  if (p->x + 25.0 < 0) cubeI--;
  if (p->y + 25.0 < 0) cubeJ--;
  if (p->z + 25.0 < 0) cubeK--;

  if (cubeI >= 0 && cubeI < kCloudDepth &&
      cubeJ >= 0 && cubeJ < kCloudWidth &&
      cubeK >= 0 && cubeK < kCloudHeight) {
    cubeInd = ToIndex(cubeI, cubeJ, cubeK);
  }
  else{
    cubeInd = 5000;
  }
  return cubeInd; 
}

size_t MapManager::find_corner_map_used(const PointType *p,int a,int b, int c)
{
  return find_map_used(p,a,b,c);
}

size_t MapManager::find_surface_map_used(const PointType *p,int a,int b, int c)
{
  return find_map_used(p,a,b,c);
}

size_t MapManager::find_none_map_used(const PointType *p,int a,int b, int c)
{
  return find_map_used(p,a,b,c); 
}

} // end of namespace lio
