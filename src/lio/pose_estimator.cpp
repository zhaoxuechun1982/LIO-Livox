#include "lio/pose_estimator.hpp"

namespace lio
{
  PoseEstimator::PoseEstimator(const float& filter_corner, const float& filter_surface)
  {
    pcf_corner_from_local_.reset(new PointCloudType);
    pcf_surface_from_local_.reset(new PointCloudType);
    pcf_none_from_local_.reset(new PointCloudType);
    laserCloudCornerLast.resize(kSlideWindowsSize);
    for(auto& p:laserCloudCornerLast)
      p.reset(new PointCloudType);
    laserCloudSurfLast.resize(kSlideWindowsSize);
    for(auto& p:laserCloudSurfLast)
      p.reset(new PointCloudType);
    laserCloudNonFeatureLast.resize(kSlideWindowsSize);
    for(auto& p:laserCloudNonFeatureLast)
      p.reset(new PointCloudType);
    pcf_corner_ptr.resize(kSlideWindowsSize);
    for(auto& p:pcf_corner_ptr)
      p.reset(new PointCloudType);
    pcf_surface_ptr.resize(kSlideWindowsSize);
    for(auto& p:pcf_surface_ptr)
      p.reset(new PointCloudType);
    pcf_none_ptr.resize(kSlideWindowsSize);
    for(auto& p:pcf_none_ptr)
      p.reset(new PointCloudType);
    pcf_corner_for_map_ptr_.reset(new PointCloudType);
    pcf_surface_for_map_ptr_.reset(new PointCloudType);
    pcf_none_for_map_ptr_.reset(new PointCloudType);
    transform_for_map_.setIdentity();
    kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeNonFeatureFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
  
    for(int i = 0; i < kLocalMapWindowSize; i++){
      localCornerMap[i].reset(new PointCloudType);
      localSurfMap[i].reset(new PointCloudType);
      localNonFeatureMap[i].reset(new PointCloudType);
    }
  
    down_size_filter_corner_.setLeafSize(filter_corner, filter_corner, filter_corner);
    downSizeFilterSurf.setLeafSize(filter_surface, filter_surface, filter_surface);
    down_size_filter_none_.setLeafSize(0.4, 0.4, 0.4);

    miu_thread_ = std::thread(&PoseEstimator::thread_map_increment_update, this);
    miu_thread_running_flag_ = true;
    map_manager_ptr_ = new MapManager(filter_corner, filter_surface);
    imu_aligner_ptr_ = ImuAligner::instance_pointer();
    lidar_feature_matcher_ptr_ = LidarFeatureMatcher::instance_pointer();
  }

  PoseEstimator::~PoseEstimator()
  {
    miu_thread_running_flag_ = false;
    if (miu_thread_.joinable()) {
      miu_thread_.join();
    }

    delete map_manager_ptr_;
    ImuAligner::destroy_instance();
    LidarFeatureMatcher::destroy_instance();
  }

  void PoseEstimator::thread_map_increment_update()
  {
    PointCloudTypePtr pcf_corner_ptr(new PointCloudType);
    PointCloudTypePtr pcf_surface_ptr(new PointCloudType);
    PointCloudTypePtr pcf_none_ptr(new PointCloudType);
    PointCloudTypePtr pcf_corner_to_map_ptr(new PointCloudType);
    PointCloudTypePtr pcf_surface_to_map_ptr(new PointCloudType);
    PointCloudTypePtr pcf_none_to_map_ptr(new PointCloudType);
    Eigen::Matrix4d transform;

    while(miu_thread_running_flag_)
    {
      std::unique_lock<std::mutex> locker(map_mutex_);
      if(!pcf_corner_for_map_ptr_->empty()) {
        map_update_id_++;
        map_manager_ptr_->feature_associate_to_map(pcf_corner_for_map_ptr_,
                                                   pcf_surface_for_map_ptr_,
                                                   pcf_none_for_map_ptr_,
                                                   pcf_corner_ptr,
                                                   pcf_surface_ptr,
                                                   pcf_none_ptr,
                                                   transform_for_map_);
        pcf_corner_for_map_ptr_->clear();
        pcf_surface_for_map_ptr_->clear();
        pcf_none_for_map_ptr_->clear();
        transform = transform_for_map_;
        locker.unlock();
  
        *pcf_corner_to_map_ptr += *pcf_corner_ptr;
        *pcf_surface_to_map_ptr += *pcf_surface_ptr;
        *pcf_none_to_map_ptr += *pcf_none_ptr;
  
        pcf_corner_ptr->clear();
        pcf_surface_ptr->clear();
        pcf_none_ptr->clear();
  
        if(map_update_id_ % kMapSkipFrameCount == 0) {
          map_manager_ptr_->map_increment_update(pcf_corner_to_map_ptr, 
                                         pcf_surface_to_map_ptr, 
                                         pcf_none_to_map_ptr,
                                         transform);
  
          pcf_corner_to_map_ptr->clear();
          pcf_surface_to_map_ptr->clear();
          pcf_none_to_map_ptr->clear();
        }
      }
      else {
        locker.unlock();
      }

      // sleep for 2ms
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  void PoseEstimator::convert_vector_to_double(const LidarFrameList& lidar_frame_list) 
  {
    int i = 0;
    for(const auto& l : lidar_frame_list) {
      Eigen::Map<Eigen::Matrix<double, 6, 1>> pr(param_p_r_[i]);
      pr.segment<3>(0) = l.p;                    
      pr.segment<3>(3) = Sophus::SO3d(l.q).log();  
  
      Eigen::Map<Eigen::Matrix<double, 9, 1>> vbias(param_v_bias_[i]);
      vbias.segment<3>(0) = l.v;
      vbias.segment<3>(3) = l.bg;
      vbias.segment<3>(6) = l.ba;
      i++;
    }
  }

  void PoseEstimator::convert_double_to_vector(LidarFrameList& lidar_frame_list)
  {
    int i = 0;
    for(auto& l : lidar_frame_list){
      Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(param_p_r_[i]);
      Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(param_v_bias_[i]);
      l.p = PR.segment<3>(0);
      l.q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
      l.v = VBias.segment<3>(0);
      l.bg = VBias.segment<3>(3);
      l.ba = VBias.segment<3>(6);
      i++;
    }
  }

void PoseEstimator::EstimateLidarPose(LidarFrameList& lidar_frame_list,
                                      const Eigen::Matrix4d& exTlb,
                                      const Eigen::Vector3d& gravity,
                                      nav_msgs::Odometry& debugInfo)                         
{
  
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  transformTobeMapped.topLeftCorner(3,3) = lidar_frame_list.back().q * exRbl;
  transformTobeMapped.topRightCorner(3,1) = lidar_frame_list.back().q * exPbl + lidar_frame_list.back().p;

  int laserCloudCornerFromMapNum = map_manager_ptr_->get_corner_map()->points.size();
  int laserCloudSurfFromMapNum = map_manager_ptr_->get_surf_map()->points.size();
  int laserCloudCornerFromLocalNum = pcf_corner_from_local_->points.size();
  int laserCloudSurfFromLocalNum = pcf_surface_from_local_->points.size();
  int stack_count = 0;
  for(const auto& l : lidar_frame_list){
    laserCloudCornerLast[stack_count]->clear();
    for(const auto& p : l.laser_cloud->points){
      if(std::fabs(p.normal_z - 1.0) < 1e-5)
        laserCloudCornerLast[stack_count]->push_back(p);
    }
    laserCloudSurfLast[stack_count]->clear();
    for(const auto& p : l.laser_cloud->points){
      if(std::fabs(p.normal_z - 2.0) < 1e-5)
        laserCloudSurfLast[stack_count]->push_back(p);
    }

    laserCloudNonFeatureLast[stack_count]->clear();
    for(const auto& p : l.laser_cloud->points){
      if(std::fabs(p.normal_z - 3.0) < 1e-5)
        laserCloudNonFeatureLast[stack_count]->push_back(p);
    }

    pcf_corner_ptr[stack_count]->clear();
    down_size_filter_corner_.setInputCloud(laserCloudCornerLast[stack_count]);
    down_size_filter_corner_.filter(*pcf_corner_ptr[stack_count]);

    pcf_surface_ptr[stack_count]->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
    downSizeFilterSurf.filter(*pcf_surface_ptr[stack_count]);

    pcf_none_ptr[stack_count]->clear();
    down_size_filter_none_.setInputCloud(laserCloudNonFeatureLast[stack_count]);
    down_size_filter_none_.filter(*pcf_none_ptr[stack_count]);
    stack_count++;
  }
  if ( ((laserCloudCornerFromMapNum >= 0 && laserCloudSurfFromMapNum > 100) || 
       (laserCloudCornerFromLocalNum >= 0 && laserCloudSurfFromLocalNum > 100))) {
    Estimate(lidar_frame_list, exTlb, gravity);
  }

  transformTobeMapped = Eigen::Matrix4d::Identity();
  transformTobeMapped.topLeftCorner(3,3) = lidar_frame_list.front().q * exRbl;
  transformTobeMapped.topRightCorner(3,1) = lidar_frame_list.front().q * exPbl + lidar_frame_list.front().p;

  std::unique_lock<std::mutex> locker(map_mutex_);
  *pcf_corner_for_map_ptr_ = *pcf_corner_ptr[0];
  *pcf_surface_for_map_ptr_ = *pcf_surface_ptr[0];
  *pcf_none_for_map_ptr_ = *pcf_none_ptr[0];
  transform_for_map_ = transformTobeMapped;
  pcf_corner_from_local_->clear();
  pcf_surface_from_local_->clear();
  pcf_none_from_local_->clear();
  update_local_map_increment(pcf_corner_for_map_ptr_, pcf_surface_for_map_ptr_, pcf_none_for_map_ptr_, transformTobeMapped);
  locker.unlock();
}

void PoseEstimator::Estimate(LidarFrameList& lidar_frame_list,
                         const Eigen::Matrix4d& exTlb,
                         const Eigen::Vector3d& gravity)
{

  int num_corner_map = 0;
  int num_surf_map = 0;

  static uint32_t frame_count = 0;
  int windowSize = lidar_frame_list.size();
  Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
  Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);
  kdtreeCornerFromLocal->setInputCloud(pcf_corner_from_local_);
  kdtreeSurfFromLocal->setInputCloud(pcf_surface_from_local_);
  kdtreeNonFeatureFromLocal->setInputCloud(pcf_none_from_local_);

  std::unique_lock<std::mutex> locker3(map_manager_ptr_->mtx_MapManager);
  for(int i = 0; i < 4851; i++){
    CornerKdMap[i] = map_manager_ptr_->getCornerKdMap(i);
    SurfKdMap[i] = map_manager_ptr_->getSurfKdMap(i);
    NonFeatureKdMap[i] = map_manager_ptr_->getNonFeatureKdMap(i);

    GlobalSurfMap[i] = map_manager_ptr_->laserCloudSurf_for_match[i];
    GlobalCornerMap[i] = map_manager_ptr_->laserCloudCorner_for_match[i];
    GlobalNonFeatureMap[i] = map_manager_ptr_->laserCloudNonFeature_for_match[i];
  }
  laserCenWidth_last = map_manager_ptr_->get_laserCloudCenWidth_last();
  laserCenHeight_last = map_manager_ptr_->get_laserCloudCenHeight_last();
  laserCenDepth_last = map_manager_ptr_->get_laserCloudCenDepth_last();

  locker3.unlock();

  // store point to line features
  std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
  for(auto& v : vLineFeatures){
    v.reserve(2000);
  }

  // store point to plan features
  std::vector<std::vector<FeaturePlaneVector>> vPlanFeatures(windowSize);
  for(auto& v : vPlanFeatures){
    v.reserve(2000);
  }

  std::vector<std::vector<FeatureNone>> vNonFeatures(windowSize);
  for(auto& v : vNonFeatures){
    v.reserve(2000);
  }

  if(windowSize == kSlideWindowsSize) {
    plan_weight_tan = 0.0003;
    thres_dist = 1.0;
  } else {
    plan_weight_tan = 0.0;
    thres_dist = 25.0;
  }

  // excute optimize process
  const int max_iters = 5;
  for(int iterOpt=0; iterOpt<max_iters; ++iterOpt) {

    convert_vector_to_double(lidar_frame_list);

    //create huber loss function
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    if(windowSize == kSlideWindowsSize) {
      loss_function = NULL;
    } else {
      loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    }
_   map_manager_ptr_->getNonFeatureKdMap(i);

    GlobalSurfMap[i] = map_manager_ptr_->laserCloudSurf_for_match[i];
    GlobalCornerMap[i] = map_manager_ptr_->laserCloudCorner_for_match[i];
    GlobalNonFeatureMap[i] = map_manager_ptr_->laserCloudNonFeature_for_match[i];
  }
  laserCenWidth_last = map_manager_ptr_->get_laserCloudCenWidth_last();
  laserCenHeight_last = map_manager_ptr_->get_laserCloudCenHeight_last();
  laserCenDepth_last = map_manager_ptr_->get_laserCloudCenDepth_last();

  locker3.unlock();

  // store point to line features
  std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
  for(auto& v : vLineFeatures){
    v.reserve(2000);
  }

  // store point to plan features
  std::vector<std::vector<FeaturePlaneVector>> vPlanFeatures(windowSize);
  for(auto& v : vPlanFeatures){
    v.reserve(2000);
  }

  std::vector<std::vector<FeatureNone>> vNonFeatures(windowSize);
  for(auto& v : vNonFeatures){
    v.reserve(2000);
  }

  if(windowSize == kSlideWindowsSize) {
    plan_weight_tan = 0.0003;
    thres_dist = 1.0;
  } else {
    plan_weight_tan = 0.0;
    thres_dist = 25.0;
  }

  // excute optimize process
  const int max_iters = 5;
  for(int iterOpt=0; iterOpt<max_iters; ++iterOpt){

    convert_vector_to_double(lidar_frame_list);

    //create huber loss function
    ceres::LossFunction* loss_function = NULL;
    loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    if(windowSize == kSlideWindowsSize) {
      loss_function = NULL;
    } else {
      loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
    }

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    for(int i=0; i<windowSize; ++i) {
      problem.AddParameterBlock(param_p_r_[i], 6);
    }

    for(int i=0; i<windowSize; ++i)
      problem.AddParameterBlock(param_v_bias_[i], 9);

    // add IMU CostFunction
    for(int f=1; f<windowSize; ++f){
      auto frame_curr = lidar_frame_list.begin();
      std::advance(frame_curr, f);
      problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imu_integrator,
                              const_cast<Eigen::Vector3d&>(gravity),
                              Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                              (frame_curr->imu_integrator.GetCovariance().inverse()).matrixL().transpose()),
                               nullptr,
                               param_p_r_[f-1],
                               param_v_bias_[f-1],
                               param_p_r_[f],
                               param_v_bias_[f]);
    }

    if (last_marginalization_info){
      // construct new marginlization_factor
      auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      problem.AddResidualBlock(marginalization_factor, nullptr,
                               last_marginalization_parameter_blocks);
    }

    Eigen::Quaterniond q_before_opti = lidar_frame_list.back().q;
    Eigen::Vector3d t_before_opti = lidar_frame_list.back().p;

    std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);
    std::vector<std::vector<ceres::CostFunction *>> edgesNon(windowSize);
    std::thread threads[3];
    for(int f=0; f<windowSize; ++f) {
      auto frame_curr = lidar_frame_list.begin();
      std::advance(frame_curr, f);
      transformTobeMapped = Eigen::Matrix4d::Identity();
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->q * exRbl;
      transformTobeMapped.topRightCorner(3,1) = frame_curr->q * exPbl + frame_curr->p;

      threads[0] = std::thread(&PoseEstimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(pcf_corner_ptr[f]),
                               std::ref(pcf_corner_from_local_),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[1] = std::thread(&PoseEstimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(pcf_surface_ptr[f]),
                               std::ref(pcf_surface_from_local_),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[2] = std::thread(&PoseEstimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(pcf_none_ptr[f]),
                               std::ref(pcf_none_from_local_),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[0].join();
      threads[1].join();
      threads[2].join();
    }

    int cntSurf = 0;
    int cntCorner = 0;
    int cntNon = 0;
    if(windowSize == kSlideWindowsSize) {
      thres_dist = 1.0;
      if(iterOpt == 0){
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }

          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }
      }else{
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(vLineFeatures[f][cntFtu].valid) {
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(vPlanFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(vNonFeatures[f][cntFtu].valid){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
            }
            cntFtu++;
            cntNon++;
          }
        }
      }
    } else {
        if(iterOpt == 0) {
          thres_dist = 10.0;
        } else {
          thres_dist = 1.0;
        }
        for(int f=0; f<windowSize; ++f){
          int cntFtu = 0;
          for (auto &e : edgesLine[f]) {
            if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }else{
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntCorner++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f]) {
            if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }else{
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntSurf++;
          }

          cntFtu = 0;
          for (auto &e : edgesNon[f]) {
            if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
              problem.AddResidualBlock(e, loss_function, param_p_r_[f]);
              vNonFeatures[f][cntFtu].valid = true;
            }else{
              vNonFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
            cntNon++;
          }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    convert_double_to_vector(lidar_frame_list);

    Eigen::Quaterniond q_after_opti = lidar_frame_list.back().q;
    Eigen::Vector3d t_after_opti = lidar_frame_list.back().p;
    Eigen::Vector3d V_after_opti = lidar_frame_list.back().v;
    double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
    double deltaT = (t_before_opti - t_after_opti).norm();

    if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt+1) == max_iters){
      ROS_INFO("Frame: %d\n",frame_count++);
      if(windowSize != kSlideWindowsSize) break;
      // apply marginalization
      auto *marginalization_info = new MarginalizationInfo();
      if (last_marginalization_info){
        std::vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
          if (last_marginalization_parameter_blocks[i] == param_p_r_[0] ||
              last_marginalization_parameter_blocks[i] == param_v_bias_[0])
            drop_set.push_back(i);
        }

        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                          last_marginalization_parameter_blocks,
                                                          drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
      
      auto frame_curr = lidar_frame_list.begin();
      std::advance(frame_curr, 1);
      ceres::CostFunction* IMU_Cost = Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                     const_cast<Eigen::Vector3d&>(gravity),
                                                                     Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                             (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                             .matrixL().transpose());
      auto *residual_block_info = new ResidualBlockInfo(IMU_Cost, nullptr,
                                                        std::vector<double *>{param_p_r_[0], param_v_bias_[0], param_p_r_[1], param_v_bias_[1]},
                                                        std::vector<int>{0, 1});
      marginalization_info->addResidualBlockInfo(residual_block_info);

      int f = 0;
      transformTobeMapped = Eigen::Matrix4d::Identity();
      transformTobeMapped.topLeftCorner(3,3) = frame_curr->q * exRbl;
      transformTobeMapped.topRightCorner(3,1) = frame_curr->q * exPbl + frame_curr->p;
      edgesLine[f].clear();
      edgesPlan[f].clear();
      edgesNon[f].clear();
      threads[0] = std::thread(&PoseEstimator::processPointToLine, this,
                               std::ref(edgesLine[f]),
                               std::ref(vLineFeatures[f]),
                               std::ref(pcf_corner_ptr[f]),
                               std::ref(pcf_corner_from_local_),
                               std::ref(kdtreeCornerFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[1] = std::thread(&PoseEstimator::processPointToPlanVec, this,
                               std::ref(edgesPlan[f]),
                               std::ref(vPlanFeatures[f]),
                               std::ref(pcf_surface_ptr[f]),
                               std::ref(pcf_surface_from_local_),
                               std::ref(kdtreeSurfFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));

      threads[2] = std::thread(&PoseEstimator::processNonFeatureICP, this,
                               std::ref(edgesNon[f]),
                               std::ref(vNonFeatures[f]),
                               std::ref(pcf_none_ptr[f]),
                               std::ref(pcf_none_from_local_),
                               std::ref(kdtreeNonFeatureFromLocal),
                               std::ref(exTlb),
                               std::ref(transformTobeMapped));      
                      
      threads[0].join();
      threads[1].join();
      threads[2].join();
      int cntFtu = 0;
      for (auto &e : edgesLine[f]) {
        if(vLineFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{param_p_r_[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }
      cntFtu = 0;
      for (auto &e : edgesPlan[f]) {
        if(vPlanFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{param_p_r_[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }

      cntFtu = 0;
      for (auto &e : edgesNon[f]) {
        if(vNonFeatures[f][cntFtu].valid){
          auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                            std::vector<double *>{param_p_r_[0]},
                                                            std::vector<int>{0});
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        cntFtu++;
      }

      marginalization_info->preMarginalize();
      marginalization_info->marginalize();

      std::unordered_map<long, double *> addr_shift;
      for (int i = 1; i < kSlideWindowsSize; i++)
      {
        addr_shift[reinterpret_cast<long>(param_p_r_[i])] = param_p_r_[i - 1];
        addr_shift[reinterpret_cast<long>(param_v_bias_[i])] = param_v_bias_[i - 1];
      }
      std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

      delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
      break;
    }

    if(windowSize != kSlideWindowsSize) {
      for(int f=0; f<windowSize; ++f){
        edgesLine[f].clear();
        edgesPlan[f].clear();
        edgesNon[f].clear();
        vLineFeatures[f].clear();
        vPlanFeatures[f].clear();
        vNonFeatures[f].clear();
      }
    }
  }

}

void PoseEstimator::update_local_map_increment(const PointCloudTypePtr& pcf_corner_ptr,
                                               const PointCloudTypePtr& pcf_surface_ptr,
                                               const PointCloudTypePtr& pcf_none_ptr,
                                               const Eigen::Matrix4d& transformTobeMapped)
{
  size_t pcf_corner_size = pcf_corner_ptr->points.size();
  size_t pcf_surface_size = pcf_surface_ptr->points.size();
  size_t pcf_none_size = pcf_none_ptr->points.size();

  PointType pointSel;
  size_t id = local_map_id_ % kLocalMapWindowSize;
  local_corner_map_[id]->clear();
  local_surface_map_[id]->clear();
  local_none_map_[id]->clear();

  for (size_t i = 0; i < pcf_corner_size; i++) {
    MapManager::point_associate_to_map(&pcf_corner_ptr->points[i], &pointSel, transformTobeMapped);
    local_corner_map_[id]->push_back(pointSel);
  }

  for (size_t i = 0; i < pcf_surface_size; i++) {
    MapManager::point_associate_to_map(&pcf_surface_ptr->points[i], &pointSel, transformTobeMapped);
    local_surface_map_[id]->push_back(pointSel);
  }
  
  for (size_t i = 0; i < pcf_none_size; i++) {
    MapManager::point_associate_to_map(&pcf_none_ptr->points[i], &pointSel, transformTobeMapped);
    local_none_map_[id]->push_back(pointSel);
  }

  for (size_t i = 0; i < kLocalMapWindowSize; i++) {
    *pcf_corner_from_local_ += *local_corner_map_[i];
    *pcf_surface_from_local_ += *local_surface_map_[i];
    *pcf_none_from_local_ += *local_none_map_[i];
  }

  PointCloudTypePtr temp(new PointCloudType());
  down_size_filter_corner_.setInputCloud(pcf_corner_from_local_);
  down_size_filter_corner_.filter(*temp);
  pcf_corner_from_local_ = temp;

  temp->clear();
  down_size_filter_surface_.setInputCloud(pcf_surface_from_local_);
  down_size_filter_surface_.filter(*temp);
  pcf_surface_from_local_ = temp;

  temp->clear();
  down_size_filter_none_.setInputCloud(pcf_none_from_local_);
  down_size_filter_none_.filter(*temp);
  pcf_none_from_local_ = temp;

  local_map_id_++;
}

bool PoseEstimator::run_imu_align(LidarFrameDeque& frame_deque,
                                  Eigen::Vector3d& g_b,
                                  const Eigen::Matrix3d& ex_r_lb, 
                                  const Eigen::Vector3d& ex_t_bl)
{
  return imu_aligner_ptr_->instance().run_align(frame_deque,g_b, ex_r_lb, ex_t_bl, kSlideWindowsSize);
}

} // end of namespace lio