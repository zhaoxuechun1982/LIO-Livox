#include "Estimator/Estimator.h"


typedef pcl::PointXYZINormal PointType;

int WINDOWSIZE;

bool g_lidar_imu_initialized_flag = false;
bool g_new_full_cloud_flag = false;

boost::shared_ptr<std::list<Estimator::LidarFrame>> g_lidar_frame_list;
pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
Estimator* g_estimator_ptr;

ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserOdometryPath;
ros::Publisher pubFullLaserCloud;
tf::StampedTransform laserOdometryTrans;
tf::TransformBroadcaster* tfBroadcaster;
ros::Publisher pubGps;



Eigen::Matrix4d transformAftMapped = Eigen::Matrix4d::Identity();

std::mutex _mutexLidarQueue;
std::queue<sensor_msgs::PointCloud2ConstPtr> _lidarMsgQueue;
std::mutex _mutexIMUQueue;
std::queue<sensor_msgs::ImuConstPtr> _imuMsgQueue;
Eigen::Matrix4d exTlb;
Eigen::Matrix3d exRlb, exRbl;
Eigen::Vector3d exPlb, exPbl;
Eigen::Vector3d g_gravity_vector;
float filter_parameter_corner = 0.2;
float filter_parameter_surf = 0.4;
int IMU_Mode = 2;
sensor_msgs::NavSatFix gps;
int g_push_count = 0;
double startTime = 0;

nav_msgs::Path laserOdoPath;

/** @brief Publish odometry infomation
  * @param[in] newPose: pose to be published
  * @param[in] timefullCloud: time stamp
  */
void publish_odometry(const Eigen::Matrix4d& newPose, double& timefullCloud)
{
  nav_msgs::Odometry laserOdometry;

  // Convert 4x4 world-lidar pose matrix to Odometry msg and publish
  Eigen::Matrix3d Rcurr = newPose.topLeftCorner(3, 3);
  Eigen::Quaterniond newQuat(Rcurr);
  Eigen::Vector3d newPosition = newPose.topRightCorner(3, 1);
  laserOdometry.header.frame_id = "/world";
  laserOdometry.child_frame_id = "/livox_frame";
  laserOdometry.header.stamp = ros::Time().fromSec(timefullCloud);
  laserOdometry.pose.pose.orientation.x = newQuat.x();
  laserOdometry.pose.pose.orientation.y = newQuat.y();
  laserOdometry.pose.pose.orientation.z = newQuat.z();
  laserOdometry.pose.pose.orientation.w = newQuat.w();
  laserOdometry.pose.pose.position.x = newPosition.x();
  laserOdometry.pose.pose.position.y = newPosition.y();
  laserOdometry.pose.pose.position.z = newPosition.z();
  pubLaserOdometry.publish(laserOdometry);

  // Append current pose to trajectory path message and publish for rviz display 
  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  laserOdoPath.header.stamp = laserOdometry.header.stamp;
  laserOdoPath.poses.push_back(laserPose);
  laserOdoPath.header.frame_id = "/world";
  pubLaserOdometryPath.publish(laserOdoPath);

  // Construct and broadcast TF transform from world to livox_frame
  laserOdometryTrans.frame_id_ = "/world";
  laserOdometryTrans.child_frame_id_ = "/livox_frame";
  laserOdometryTrans.stamp_ = ros::Time().fromSec(timefullCloud);
  laserOdometryTrans.setRotation(tf::Quaternion(newQuat.x(), newQuat.y(), newQuat.z(), newQuat.w()));
  laserOdometryTrans.setOrigin(tf::Vector3(newPosition.x(), newPosition.y(), newPosition.z()));
  tfBroadcaster->sendTransform(laserOdometryTrans);

  // Generate fake GPS message using world coordinate for third-party component compatibility
	gps.header.stamp = ros::Time().fromSec(timefullCloud);
	gps.header.frame_id = "world";
	gps.latitude = newPosition.x();
	gps.longitude = newPosition.y();
	gps.altitude = newPosition.z();
	gps.position_covariance = { Rcurr(0, 0), Rcurr(1, 0), Rcurr(2, 0),
                              Rcurr(0, 1), Rcurr(1, 1), Rcurr(2, 1),
                              Rcurr(0, 2), Rcurr(1, 2), Rcurr(2, 2)};
	pubGps.publish(gps);
}

/** @brief Lidar point cloud callback
  * @param[in] msg: pose message received 
  * @note buffer message into queue with mutex protection
  */
void full_point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  // push lidar msg to queue
	std::unique_lock<std::mutex> lock(_mutexLidarQueue);
  _lidarMsgQueue.push(msg);
}

/** @brief IMU data callback
  * @param[in] imu_msg: imu_msg message received
  * @note buffer message into queue with mutex protection
  */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
  // push IMU msg to queue
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  _imuMsgQueue.push(imu_msg);
}

/** @brief Data time series boundary alignment
  * @param[in] startTime: left boundary of time interval
  * @param[in] endTime: right boundary of time interval
  * @param[in] vimuMsg: store IMU messages
  * @note Fetch IMU data within [startTime, endTime] for lidar frame pre-integration, 
  *       linear interpolate if time boundary missing
  */
bool fetch_imu_msgs(double startTime, double endTime, std::vector<sensor_msgs::ImuConstPtr> &vimuMsg) 
{
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  double current_time = 0;
  vimuMsg.clear();
  while(true)
  {
    if(_imuMsgQueue.empty())
      break;
    if(_imuMsgQueue.back()->header.stamp.toSec()<endTime ||
       _imuMsgQueue.front()->header.stamp.toSec()>=endTime)
      break;
    sensor_msgs::ImuConstPtr& tmpimumsg = _imuMsgQueue.front();
    double time = tmpimumsg->header.stamp.toSec();
    if(time<=endTime && time>startTime){
      vimuMsg.push_back(tmpimumsg);
      current_time = time;
      _imuMsgQueue.pop();
      if(time == endTime) break;
    } else{
      if(time<=startTime){
        _imuMsgQueue.pop();
      } else{
        double dt_1 = endTime - current_time;
        double dt_2 = time - endTime;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        sensor_msgs::ImuPtr theLastIMU(new sensor_msgs::Imu);
        theLastIMU->linear_acceleration.x = w1 * vimuMsg.back()->linear_acceleration.x + w2 * tmpimumsg->linear_acceleration.x;
        theLastIMU->linear_acceleration.y = w1 * vimuMsg.back()->linear_acceleration.y + w2 * tmpimumsg->linear_acceleration.y;
        theLastIMU->linear_acceleration.z = w1 * vimuMsg.back()->linear_acceleration.z + w2 * tmpimumsg->linear_acceleration.z;
        theLastIMU->angular_velocity.x = w1 * vimuMsg.back()->angular_velocity.x + w2 * tmpimumsg->angular_velocity.x;
        theLastIMU->angular_velocity.y = w1 * vimuMsg.back()->angular_velocity.y + w2 * tmpimumsg->angular_velocity.y;
        theLastIMU->angular_velocity.z = w1 * vimuMsg.back()->angular_velocity.z + w2 * tmpimumsg->angular_velocity.z;
        theLastIMU->header.stamp.fromSec(endTime);
        vimuMsg.emplace_back(theLastIMU);
        break;
      }
    }
  }

  return !vimuMsg.empty();
}

/** @brief Remove Lidar Distortion
  * @param[in] cloud: lidar cloud need to be undistorted
  * @param[in] rm_01: delta rotation from frame-start lidar to frame-end lidar
  * @param[in] tv_01: delta translation from frame-start lidar to frame-end lidar
  */
void remove_lidar_distortion(pcl::PointCloud<PointType>::Ptr& cloud,
                             const Eigen::Matrix3d& rm_01, 
                             const Eigen::Vector3d& tv_01)
{
  for (int i = 0; i < cloud->points.size(); i++) {

    // s: normalized sample time of current point [0 = frame start, 1 = frame end]
    float s = cloud->points[i].normal_x;

    // Quaternion for rm_01
    Eigen::Quaterniond q_01 = Eigen::Quaterniond(rm_01).normalized();

    // Spherical linear interpolation: rotation from L0 to instantaneous scan time s
    Eigen::Quaterniond q_0s = Eigen::Quaterniond::Identity().slerp(s, q_01).normalized();
    
    // Linear interpolation: translation from L0 to instantaneous scan time s
    const Eigen::Vector3d tv_0s = s * tv_01;
    
    // Step1: Transform raw point from instantaneous scan pose(s) back to frame-start L0 coordinate
    Eigen::Vector3d p_0 = q_0s * Eigen::Vector3d(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z) + tv_0s;
    
    // Step2: Inverse transform from L0 to frame-end L1 coordinate, eliminate scan distortion
    Eigen::Vector3d p_1 = rm_01.transpose() * (p_s0 - tv_01);

    // Overwrite original point with undistorted coordinate
    cloud->points[i].x = p_1(0);
    cloud->points[i].y = p_1(1);
    cloud->points[i].z = p_1(2);

    // Mark this point fully aligned to end-of-frame timestamp
    cloud->points[i].normal_x = 1.0;
  }
}

bool TryMAPInitialization(void)
{
  Eigen::Vector3d average_acc = -g_lidar_frame_list->begin()->imuIntegrator.GetAverageAcc();
  double info_g = std::fabs(9.805 - average_acc.norm());
  average_acc = average_acc * 9.805 / average_acc.norm();

  // calculate the initial gravity direction
  double para_quat[4];
  para_quat[0] = 1;
  para_quat[1] = 0;
  para_quat[2] = 0;
  para_quat[3] = 0;


  ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
  ceres::Problem problem_quat;
  
  problem_quat.AddParameterBlock(para_quat, 4, quatParam);

  problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc),
                                nullptr,
                                para_quat);

  ceres::Solver::Options options_quat;
  ceres::Solver::Summary summary_quat;
  ceres::Solve(options_quat, &problem_quat, &summary_quat);

  Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2], para_quat[3]);


  //build prior factor of LIO initialization
  Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> prior_v;
  
  /*************************************************** */
  Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix());
  prior_r = SO3_R_wg.log();
  /*************************************************** */
  int v_size = g_lidar_frame_list->size();
  for(int i = 0; i < v_size; i++) {
    prior_v.push_back(Eigen::Vector3d::Zero());
  }
  for (int i = 1; i < v_size; i++){
    auto iter = g_lidar_frame_list->begin();
    auto iter_next = g_lidar_frame_list->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    Eigen::Vector3d velo_imu = (iter_next->P - iter->P + iter_next->Q*exPlb - iter->Q*exPlb) / (iter_next->timeStamp - iter->timeStamp);
    prior_v[i] = velo_imu;
  }
  prior_v[0] = prior_v[1];
  /*************************************************** */
  double para_v[v_size][3];
  double para_r[3];
  double para_ba[3];
  double para_bg[3];

  for(int i = 0; i < 3; i++) {
    para_r[i] = 0;
    para_ba[i] = 0;
    para_bg[i] = 0;
  }

  for(int i = 0; i < v_size; i++) {
    for(int j = 0; j < 3; j++) {
      para_v[i][j] = prior_v[i][j];
    }
  }

  Eigen::Matrix<double, 3, 3> sqrt_information_r = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_v = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(para_r, 3);
  problem.AddParameterBlock(para_ba, 3);
  problem.AddParameterBlock(para_bg, 3);
  for(int i = 0; i < v_size; i++) {
    problem.AddParameterBlock(para_v[i], 3);
  }
  
  // add CostFunction
  problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
                           nullptr,
                           para_r);
  
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
                           nullptr,
                           para_ba);
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
                           nullptr,
                           para_bg);

  for(int i = 0; i < v_size; i++) {
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v[i], sqrt_information_v),
                             nullptr,
                             para_v[i]);
  }

  for(int i = 1; i < v_size; i++) {
    auto iter = g_lidar_frame_list->begin();
    auto iter_next = g_lidar_frame_list->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    Eigen::Vector3d pi = iter->P + iter->Q*exPlb;
    Sophus::SO3d SO3_Ri(iter->Q*exRlb);
    Eigen::Vector3d ri = SO3_Ri.log();
    Eigen::Vector3d pj = iter_next->P + iter_next->Q*exPlb;
    Sophus::SO3d SO3_Rj(iter_next->Q*exRlb);
    Eigen::Vector3d rj = SO3_Rj.log();

    problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                                                   ri,
                                                                   rj,
                                                                   pj-pi,
                                                                   Eigen::LLT<Eigen::Matrix<double, 9, 9>>
                                                                    (iter_next->imuIntegrator.GetCovariance().block<9,9>(0,0).inverse())
                                                                    .matrixL().transpose()),
                             nullptr,
                             para_r,
                             para_v[i-1],
                             para_v[i],
                             para_ba,
                             para_bg);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 6;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
  g_gravity_vector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

  Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
  Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);

  if(ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5) {
    ROS_WARN("Too Large Biases! Initialization Failed!");
    return false;
  }

  for(int i = 0; i < v_size; i++) {
    auto iter = g_lidar_frame_list->begin();
    std::advance(iter, i);
    iter->ba = ba_vec;
    iter->bg = bg_vec;
    Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
    if((bv_vec - prior_v[i]).norm() > 2.0) {
      ROS_WARN("Too Large Velocity! Initialization Failed!");
      std::cout<<"delta v norm: "<<(bv_vec - prior_v[i]).norm()<<std::endl;
      return false;
    }
    iter->V = bv_vec;
  }
/**************************************************************************/
  for(size_t i = 0; i < v_size - 1; i++) {
    auto laser_trans_i = g_lidar_frame_list->begin();
    auto laser_trans_j = g_lidar_frame_list->begin();
    std::advance(laser_trans_i, i);
    std::advance(laser_trans_j, i+1);
    laser_trans_j->imuIntegrator.PreIntegration(laser_trans_i->timeStamp, laser_trans_i->bg, laser_trans_i->ba);
  }


  // //if IMU success initialized
  WINDOWSIZE = Estimator::SLIDEWINDOWSIZE;
  while(g_lidar_frame_list->size() > WINDOWSIZE) {
	  g_lidar_frame_list->pop_front();
  }
	Eigen::Vector3d Pwl = g_lidar_frame_list->back().P;
	Eigen::Quaterniond Qwl = g_lidar_frame_list->back().Q;
	g_lidar_frame_list->back().P = Pwl + Qwl*exPlb;
	g_lidar_frame_list->back().Q = Qwl * exRlb;

	// std::cout << "\n=============================\n| Initialization Successful |"<<"\n=============================\n" << std::endl;
  
  return true;
}

/** \brief Mapping main thread
  */
void process(void)
{
  double time_last_lidar = -1;                             // Timestamp of previous lidar frame
  double time_curr_lidar = -1;                             // Timestamp of current incoming lidar frame
  Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();  // Rotation matrix transform from previous lidar frame to current lidar frame (lidar coordinate system)
  Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();      // Frame-to-frame translation transform from previous lidar frame to current lidar frame (lidar coordinate system)
	Eigen::Matrix3d delta_Rb = Eigen::Matrix3d::Identity();  // Frame-to-frame rotation transform from previous IMU body frame to current IMU body frame (body coordinate system)
	Eigen::Vector3d delta_tb = Eigen::Vector3d::Zero();      // Frame-to-frame translation transform from previous IMU body frame to current IMU body frame (body coordinate system)
  std::vector<sensor_msgs::ImuConstPtr> vimuMsg;           // Buffer to store IMU measurements between two consecutive lidar frames
  
  while(ros::ok()) 
  {
    g_new_full_cloud_flag = false;                              // Flag indicating whether new lidar frame is fetched
    laserCloudFullRes.reset(new pcl::PointCloud<PointType>());  // Allocate empty point cloud buffer for current frame

    // Lock queue to avoid concurrent access with lidar callback thread 
	  std::unique_lock<std::mutex> lock_lidar(_mutexLidarQueue);
    if(!_lidarMsgQueue.empty()) {
      // Record timestamp of current lidar frame
      time_curr_lidar = _lidarMsgQueue.front()->header.stamp.toSec();
      // Convert ROS point cloud message to PCL format
      pcl::fromROSMsg(*_lidarMsgQueue.front(), *laserCloudFullRes);
      // Discard processed lidar message from queue
      _lidarMsgQueue.pop();

      // Mark valid lidar frame received
      g_new_full_cloud_flag = true;
    }
    lock_lidar.unlock();

    // Process pipeline only if new lidar frame exists
    if(g_new_full_cloud_flag) {
      // Debug odometry placeholder for g_estimator_ptr output
      nav_msgs::Odometry debugInfo;
      debugInfo.pose.pose.position.x = 0;
      debugInfo.pose.pose.position.y = 0;
      debugInfo.pose.pose.position.z = 0;

      // Fetch IMU measurements between previous lidar frame and current frame if IMU enabled
      if(IMU_Mode > 0 && time_last_lidar > 0) {
        // get IMU msg int the Specified time interval
        vimuMsg.clear();
        int countFail = 0;

        // Retry until IMU data is ready or reach max retry limit
        while (!fetch_imu_msgs(time_last_lidar, time_curr_lidar, vimuMsg)) {
          countFail++;
          if (countFail > 100){
            break;
          }

          // Wait 10ms for IMU buffer to fill
          std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }
      }

      Estimator::LidarFrame lidarFrame;           // Frame container to store current lidar frame information
      lidarFrame.laserCloud = laserCloudFullRes;  // Attach undistorted raw point cloud to frame
      lidarFrame.timeStamp = time_curr_lidar;     // Record absolute timestamp of current lidar frame

      // Smart pointer holding frame list passed to pose g_estimator_ptr
	    boost::shared_ptr<std::list<Estimator::LidarFrame>> lidar_list; 
	    
      // Branch 1: Valid IMU measurements between two lidar frames exist
      if(!vimuMsg.empty()) {
        // Sub-branch 1.1: System not finished IMU initialization
	    	if(!g_lidar_imu_initialized_flag) {
	    		// Feed IMU samples into integrator buffer
			    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
          // Only integrate gyroscope to get relative rotation (no acceleration pre-integration)
			    lidarFrame.imuIntegrator.GyroIntegration(time_last_lidar);
          // Extract body-frame relative rotation from integrator
			    delta_Rb = lidarFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();
          // Convert body-frame delta rotation to lidar-frame via extrinsic Tlb
			    delta_Rl = exTlb.topLeftCorner(3, 3) * delta_Rb * exTlb.topLeftCorner(3, 3).transpose();

			    // predict current lidar pose
			    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
			                 + transformAftMapped.topRightCorner(3,1);
			    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
			    lidarFrame.Q = m3d;

          // Create temporary single-frame list for g_estimator_ptr input
			    lidar_list.reset(new std::list<Estimator::LidarFrame>);
			    lidar_list->push_back(lidarFrame);
		    }
        // Sub-branch 1.2: IMU initialization complete, enable full IMU pre-integration
        else {
			    // Buffer all IMU samples into integrator
			    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
          // Execute full pre-integration using previous frame's bias bg/ba
			    lidarFrame.imuIntegrator.PreIntegration(g_lidar_frame_list->back().timeStamp, g_lidar_frame_list->back().bg, g_lidar_frame_list->back().ba);

          // Fetch world state of previous frame in sliding window
			    const Eigen::Vector3d& Pwbpre = g_lidar_frame_list->back().P;
			    const Eigen::Quaterniond& Qwbpre = g_lidar_frame_list->back().Q;
			    const Eigen::Vector3d& Vwbpre = g_lidar_frame_list->back().V;

          // Get pre-integrated increments in body frame
			    const Eigen::Quaterniond& dQ =  lidarFrame.imuIntegrator.GetDeltaQ();
			    const Eigen::Vector3d& dP = lidarFrame.imuIntegrator.GetDeltaP();
			    const Eigen::Vector3d& dV = lidarFrame.imuIntegrator.GetDeltaV();
			    double dt = lidarFrame.imuIntegrator.GetDeltaTime();

          // Predict current body world rotation, position, velocity via pre-integration result
			    lidarFrame.Q = Qwbpre * dQ;
			    lidarFrame.P = Pwbpre + Vwbpre*dt + 0.5*g_gravity_vector*dt*dt + Qwbpre*(dP);
			    lidarFrame.V = Vwbpre + g_gravity_vector*dt + Qwbpre*(dV);

          // Inherit IMU bias from last frame (bias stays constant within one window)
			    lidarFrame.bg = g_lidar_frame_list->back().bg;
			    lidarFrame.ba = g_lidar_frame_list->back().ba;

          // Calculate world-to-lidar transform of previous frame
			    Eigen::Quaterniond Qwlpre = Qwbpre * Eigen::Quaterniond(exRbl);
			    Eigen::Vector3d Pwlpre = Qwbpre * exPbl + Pwbpre;

          // Calculate predicted world-to-lidar transform of current frame
			    Eigen::Quaterniond Qwl = lidarFrame.Q * Eigen::Quaterniond(exRbl);
			    Eigen::Vector3d Pwl = lidarFrame.Q * exPbl + lidarFrame.P;

          // Compute frame-to-frame relative transform in lidar coordinate (L_prev -> L_curr)
			    delta_Rl = Qwlpre.conjugate() * Qwl;
			    delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);

          // Cache body-frame pre-integrated increments for next frame rough prediction
			    delta_Rb = dQ.toRotationMatrix();
			    delta_tb = dP;

          // Maintain fixed-size sliding window: push new frame, pop oldest frame
			    g_lidar_frame_list->push_back(lidarFrame);
			    g_lidar_frame_list->pop_front();

          // Pass sliding window list to g_estimator_ptr
			    lidar_list = g_lidar_frame_list;
	    	}
	    }
      
      // Branch 2: No IMU data available between two lidar frames
      else {
        // If system already initialized, skip current frame (cannot compute valid motion delta)
	    	if(g_lidar_imu_initialized_flag) {
	    	  break;
        }
	    	else {
			    // Early stage without IMU data: reuse historical delta to predict pose roughly
			    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
			                 + transformAftMapped.topRightCorner(3,1);
			    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
			    lidarFrame.Q = m3d;

          // Temporary single-frame list
			    lidar_list.reset(new std::list<Estimator::LidarFrame>);
			    lidar_list->push_back(lidarFrame);
	    	}
	    }

	    // Remove lidar distortion
	    remove_lidar_distortion(laserCloudFullRes, delta_Rl, delta_tl);

      // Optimize current lidar pose with IMU
      g_estimator_ptr->EstimateLidarPose(*lidar_list, exTlb, g_gravity_vector, debugInfo);

      // Pointer buffer for corner and surface feature map points
      pcl::PointCloud<PointType>::Ptr laserCloudCornerMap(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr laserCloudSurfMap(new pcl::PointCloud<PointType>());

      // 4x4 extrinsic pose matrix of lidar in world frame, initialized as identity
	    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
      // R_WL = R_WB * R_BL
	    transformTobeMapped.topLeftCorner(3,3) = lidar_list->front().Q * exRbl;
      // P_WL = R_WB * P_BL + P_WB
	    transformTobeMapped.topRightCorner(3,1) = lidar_list->front().Q * exPbl + lidar_list->front().P;

	    // Update body-frame relative motion from last frame to current frame
	    delta_Rb = transformAftMapped.topLeftCorner(3, 3).transpose() * lidar_list->front().Q.toRotationMatrix();
	    delta_tb = transformAftMapped.topLeftCorner(3, 3).transpose() * (lidar_list->front().P - transformAftMapped.topRightCorner(3, 1));
	    
      // Last frame lidar pose in world frame T_WL_prev
      Eigen::Matrix3d Rwlpre = transformAftMapped.topLeftCorner(3, 3) * exRbl;
	    Eigen::Vector3d Pwlpre = transformAftMapped.topLeftCorner(3, 3) * exPbl + transformAftMapped.topRightCorner(3, 1);
	    
      // Frame-to-frame lidar relative transform L_prev -> L_curr
      delta_Rl = Rwlpre.transpose() * transformTobeMapped.topLeftCorner(3,3);
	    delta_tl = Rwlpre.transpose() * (transformTobeMapped.topRightCorner(3,1) - Pwlpre);

      // Overwrite global cached pose with current optimized body world pose, as previous frame for next iteration
	    transformAftMapped.topLeftCorner(3,3) = lidar_list->front().Q.toRotationMatrix();
	    transformAftMapped.topRightCorner(3,1) = lidar_list->front().P;

	    // publish odometry rostopic
	    publish_odometry(transformTobeMapped, lidar_list->front().timeStamp);

      // Publish lidar points
      // Get total point count of undistorted lidar frame (lidar frame coordinate)
      int laserCloudFullResNum = lidar_list->front().laserCloud->points.size();
      // Create new point cloud buffer to store points transformed to world frame
      pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(new pcl::PointCloud<PointType>());
      // Pre-allocate memory to avoid repeated reallocation in loop
      laserCloudAfterEstimate->reserve(laserCloudFullResNum);

      // Transform every point from lidar frame to world frame
      for (int i = 0; i < laserCloudFullResNum; i++) {
        PointType temp_point;
        // Apply T_WL transform: lidar coordinate point -> world coordinate point
        MAP_MANAGER::pointAssociateToMap(&lidar_list->front().laserCloud->points[i], &temp_point, transformTobeMapped);
        laserCloudAfterEstimate->push_back(temp_point);
      }

      // Convert PCL point cloud to ROS message format for publishing
      sensor_msgs::PointCloud2 laserCloudMsg;
      pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
      // Mark output cloud under world coordinate system
      laserCloudMsg.header.frame_id = "/world";
      // Assign timestamp same as current lidar frame
      laserCloudMsg.header.stamp.fromSec(lidar_list->front().timeStamp);
      // Publish global aligned point cloud to ROS topic
      pubFullLaserCloud.publish(laserCloudMsg);

	    // Only run MAP initialization logic in tight-coupled mode before system initialization
	    if(IMU_Mode > 1 && !g_lidar_imu_initialized_flag) {
		    // Overwrite current frame's world pose to lidar world pose T_WL from optimized result
		    lidarFrame.P = transformTobeMapped.topRightCorner(3,1);
		    Eigen::Matrix3d m3d = transformTobeMapped.topLeftCorner(3,3);
		    lidarFrame.Q = m3d;

		    // static int g_push_count = 0;

        // Static counter to control pre-integration update frequency
		    if(g_push_count == 0) {
          // First trigger: push brand new frame into sliding window
			    g_lidar_frame_list->push_back(lidarFrame);
          // Clear historical IMU buffer of new frame's integrator
			    g_lidar_frame_list->back().imuIntegrator.Reset();
          // Maintain fixed window size, discard oldest frame if overflow
			    if(g_lidar_frame_list->size() > WINDOWSIZE) {
            g_lidar_frame_list->pop_front();
          }
		    }
        // Not first trigger: update the last frame in window with latest data
        else {
			    g_lidar_frame_list->back().laserCloud = lidarFrame.laserCloud;
			    g_lidar_frame_list->back().imuIntegrator.PushIMUMsg(vimuMsg);
			    g_lidar_frame_list->back().timeStamp = lidarFrame.timeStamp;
			    g_lidar_frame_list->back().P = lidarFrame.P;
			    g_lidar_frame_list->back().Q = lidarFrame.Q;
		    }

        // Increment counter for frequency control
		    g_push_count++;

		    if (g_push_count >= 3) {
			    g_push_count = 0;
          // If window contains at least two frames, recompute IMU pre-integration between last two frames
			    if(g_lidar_frame_list->size() > 1) {
				    auto iterRight = std::prev(g_lidar_frame_list->end());
				    auto iterLeft = std::prev(std::prev(g_lidar_frame_list->end()));
				    iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
			    }

          // Record start timestamp when window reaches partial threshold
          if (g_lidar_frame_list->size() == int(WINDOWSIZE / 1.5)) {
					  startTime = g_lidar_frame_list->back().timeStamp;
			    }

          // Trigger MAP initialization when sliding window is fully filled and time range meets requirement
			    if (!g_lidar_imu_initialized_flag 
            && g_lidar_frame_list->size() == WINDOWSIZE 
            && g_lidar_frame_list->front().timeStamp >= startTime) {
            std::cout<<"**************Start MAP Initialization!!!******************"<<std::endl;
				    // Execute multi-frame joint initialization to solve gravity, IMU bias, initial pose
            //if (g_estimator_ptr->run_imu_align(g_lidar_frame_list, g_gravity_vector, exRlb, exPlb))
            //{
            //  g_gravity_vector = temp_g;
            //  g_lidar_imu_initialized_flag = true;
            //  g_push_count = 0;
            //  startTime = 0;              
            //}
            if(TryMAPInitialization()) {
              g_lidar_imu_initialized_flag = true;
					    g_push_count = 0;
              startTime = 0;
				    }
            std::cout<<"**************Finish MAP Initialization!!!******************"<<std::endl;
			    }
		    }
	    }

      time_last_lidar = time_curr_lidar;
    } // end of if(g_new_full_cloud_flag)
  } // end of while loop
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "PoseEstimation");
  ros::NodeHandle nodeHandler("~");

  ros::param::get("~filter_parameter_corner",filter_parameter_corner);
  ros::param::get("~filter_parameter_surf",filter_parameter_surf);
  ros::param::get("~IMU_Mode",IMU_Mode);
	std::vector<double> vecTlb;
	ros::param::get("~Extrinsic_Tlb",vecTlb);

  // set extrinsic matrix between lidar & IMU
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
	R << vecTlb[0], vecTlb[1], vecTlb[2],
	     vecTlb[4], vecTlb[5], vecTlb[6],
	     vecTlb[8], vecTlb[9], vecTlb[10];
	t << vecTlb[3], vecTlb[7], vecTlb[11];
  Eigen::Quaterniond qr(R);
  R = qr.normalized().toRotationMatrix();
  exTlb.topLeftCorner(3,3) = R;
  exTlb.topRightCorner(3,1) = t;
  exRlb = R;
  exRbl = R.transpose();
  exPlb = t;
  exPbl = -1.0 * exRbl * exPlb;

  // Subscribe to the distorted point cloud
  ros::Subscriber subFullCloud = nodeHandler.subscribe<sensor_msgs::PointCloud2>("/livox_full_cloud", 10, full_point_cloud_callback);
  
  // Create an IMU subscription condition, IMU_Mode = 0 means no IMU data
  ros::Subscriber sub_imu;
  if(IMU_Mode > 0) {
    sub_imu = nodeHandler.subscribe("/livox/imu", 2000, imu_callback, ros::TransportHints().unreliable());
  }

  // Configure the size of the sliding window
  if (IMU_Mode < 2) {
    WINDOWSIZE = 1;
  } 
  else {
    WINDOWSIZE = 20;
  }

  // Publish full point cloud transformed to world frame
  pubFullLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud_mapped", 10);
  // Publish 6DoF lidar odometry in world coordinate
  pubLaserOdometry = nodeHandler.advertise<nav_msgs::Odometry> ("/livox_odometry_mapped", 5);
  // Publish accumulated trajectory path for rviz visualization
  pubLaserOdometryPath = nodeHandler.advertise<nav_msgs::Path> ("/livox_odometry_path_mapped", 5);
  // Publish fake GPS topic for third-party module compatibility
	pubGps = nodeHandler.advertise<sensor_msgs::NavSatFix>("/lidar", 1000);

  // Create tf broadcaster to publish world -> livox_frame transform (sync on caller thread)
  tfBroadcaster = new tf::TransformBroadcaster();
  // Allocate buffer to store single frame undistorted point cloud
  laserCloudFullRes.reset(new pcl::PointCloud<PointType>);
  // Initialize LIO sliding window g_estimator_ptr with feature weight params
  g_estimator_ptr = new Estimator(filter_parameter_corner, filter_parameter_surf);
  // Create sliding window list to cache historical lidar frames with IMU states
	g_lidar_frame_list.reset(new std::list<Estimator::LidarFrame>);

  // Create and immediately launch background thread for SLAM computation
  std::thread thread_process{process};
  thread_process.detach();

  ros::spin();

  return 0;
}
