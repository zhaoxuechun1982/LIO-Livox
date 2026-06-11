#ifndef _COMMON_HPP
#define _CONNON_HPP

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

using namespace std;

typedef struct
{
    Eigen::Matrix3f eigenVectorsPCA;
    Eigen::Vector3f eigenValuesPCA;
    std::vector<int> neibors;
} SNeiborPCA_cor;

int GetNeiborPCA_cor(SNeiborPCA_cor &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius);
/**
 * @brief 地面点筛选函数
 * @details 将输入点云划分成2D栅格，统计每个栅格的高度信息，筛选满足以下条件的点作为地面点：
 *          1. 栅格内最高点与平均高度差 < 0.4m（平坦区域）
 *          2. 栅格内点数量 > 3个
 *          3. 栅格平均高度 < 2m（低矮区域，代表地面）
 * @param outPoints 输出参数，筛选后的地面点云（格式：x,y,z,intensity 连续排列）
 * @param inPoints  输入参数，待筛选的原始点云（格式：x,y,z,intensity 连续排列）
 * @param inNum     输入参数，输入点云的总点数
 *
 * @return int 成功筛选出的地面点数量
 *
 * @note 通过栅格高度统计，筛选出平坦、低矮的地面候选点
 * @note 栅格范围：X[-20,20]m，Y[-10,10]m，栅格大小 2m×2m
 * @note 输入点云格式为 float[4] 连续存储：(x,y,z,i)
 * @note 方法本质上是一种前置初筛
 */
int FilterGndForPos_cor(float* outPoints, float*inPoints, int inNum);

/**
 * @brief 通过邻域PCA分片拟合地面平面，输出地面平均法向量与平面中心点。
 *
 * 流程：
 * 1. 将外部浮点数组点云转为PCL点云结构
 * 2. KdTree半径邻域搜索，逐块执行局部PCA形态分析
 * 3. 利用特征值比值筛选平坦地面面片，统一法向量朝上
 * 4. 累加有效面片的法向与采样中心，求全局均值
 * 5. 对法向量X/Y分量做分段非线性矫正补偿
 *
 * @param[out] gnd 输出数组，长度固定6
 *         gnd[0],gnd[1],gnd[2] = 地面平均法向量(x,y,z)
 *         gnd[3],gnd[4],gnd[5] = 地面面片平均中心点(x,y,z)
 * @param[in] fPoints 输入原始点云浮点数组，存储格式：[x0,y0,z0,unused, x1,y1,z1,unused,...]
 *                    每4个float为一组点数据，第4位占位无实际意义
 * @param[in] pointNum 输入点云总数量
 * @param[in] fSearchRadius KdTree邻域搜索半径，单位m
 * @return int 有效地面面片采样数量nNum；若有效面片为0，gnd保持初始0值
 *
 * @note 判定平面条件：npca.eigenValuesPCA[1]/(npca.eigenValuesPCA[0] + DIV_EPS) > PCA_PLANE_RATIO_THR
 * @note 法向定向规则：若PCA主方向Z分量向下，则整体取反保证法向量朝上
 * @note 内部依赖：GetNeiborPCA_cor()、SNeiborPCA_cor 邻域PCA计算结构体与函数
 * @warning
 * 1. gnd外部调用需保证分配至少6个float空间，否则存在数组越界风险
 * 2. pointNum小于等于MIN_VALID_POINTS(3)时直接返回0，无结果输出
 * 3. 最大采样面片数量限制MAX_SAMPLE_PATCH(1000)，达到上限停止遍历剩余点
 * 4. 内存使用calloc分配标记数组pLabel，函数内已配套free释放
 */
int CalGndPos_cor(float *gnd, float *fPoints, int pointNum, float fSearchRadius);
int GetRTMatrix_cor(float *RTM, float *v0, float *v1);
int CorrectPoints_cor(float *fPoints,int pointNum,float *gndPos);
/**
 * @brief 外层入口：筛选地面点并平滑更新全局地面平面参数
 *
 * 整体执行流程：
 * 1. 预分配缓存数组，调用FilterGndForPos_cor从全部点云中过滤出候选地面点
 * 2. 将地面候选点送入CalGndPos_cor，通过邻域PCA计算当前帧地面临时平面（法向量+中心坐标）
 * 3. 基于全局缓存gnd_pos做时序平滑更新逻辑：
 *    - 全局无初始地面参数：直接赋值当前帧结果
 *    - 全局已有参数：帧数未超限且法向量偏差小时取新旧均值平滑；偏差大/帧数超限则直接覆盖刷新
 * 4. 将最终平滑后的全局地面参数拷贝输出到pos数组，释放临时点云缓存内存
 *
 * @param[out] pos 输出长度为6的浮点数组
 *         pos[0]/pos[1]/pos[2]：平滑后地面平均法向量(x,y,z)
 *         pos[3]/pos[4]/pos[5]：平滑后地面面片平均中心点(x,y,z)
 * @param[in] fPoints 输入完整点云浮点数组，四点一组[x,y,z,占位]
 * @param[in] pointNum 输入总点云数量
 * @return int 固定返回0，无失败错误码区分
 *
 * @note 依赖全局变量：
 *       gnd_pos[6]：全局持久存储的地面平面参数
 *       frame_count：连续未刷新计数
 *       frame_lenth_threshold：连续帧数阈值
 * @note 固定配置硬编码：
 *       临时地面点最大容量60000点；CalGndPos_cor固定搜索半径1.0m；法向差异阈值0.1
 * @note 内部依赖子函数：FilterGndForPos_cor、CalGndPos_cor
 *
 * @warning
 * 1. pos必须外部保证至少6个float空间，否则越界
 * 2. 使用calloc手动分配fPoints3，函数内配套free释放；异常分支存在内存泄漏风险
 * 3. 无指针判空：若fPoints为空、pointNum非法会直接内存访问崩溃
 * 4. 返回值恒为0，无法通过返回值识别地面点过少、PCA拟合失败等异常状态
 * 5. 平滑逻辑耦合全局状态，不支持多实例并行运行
 */
int GetGndPos(float *pos, float *fPoints,int pointNum);

#endif
