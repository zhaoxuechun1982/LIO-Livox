#ifndef _SEGMENT_HPP
#define _SEGMENT_HPP

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

#include "pointsCorrect.hpp"

using namespace std;

#define SELF_CALI_FRAMES 20

#define GND_IMG_NX 150
#define GND_IMG_NY 400
#define GND_IMG_DX 0.2
#define GND_IMG_DY 0.2
#define GND_IMG_OFFX 40
#define GND_IMG_OFFY 40

#define GND_IMG_NX1 24
#define GND_IMG_NY1 20
#define GND_IMG_DX1 4
#define GND_IMG_DY1 4
#define GND_IMG_OFFX1 40
#define GND_IMG_OFFY1 40

#define DN_SAMPLE_IMG_NX 600 //(GND_IMG_NX)
#define DN_SAMPLE_IMG_NY 200 //(GND_IMG_NY)
#define DN_SAMPLE_IMG_NZ 100 //(GND_IMG_NZ) 
#define DN_SAMPLE_IMG_DX 0.4 //(GND_IMG_DX)
#define DN_SAMPLE_IMG_DY 0.4 //(GND_IMG_DY)
#define DN_SAMPLE_IMG_DZ 0.2
#define DN_SAMPLE_IMG_OFFX 40   //(GND_IMG_OFFX)
#define DN_SAMPLE_IMG_OFFY 40   //(GND_IMG_OFFY)
#define DN_SAMPLE_IMG_OFFZ 2.5  //2.5

#define FREE_ANG_NUM 360
#define FREE_PI (3.14159265)
#define FREE_DELTA_ANG (FREE_PI*2/FREE_ANG_NUM)

typedef struct
{
    Eigen::Matrix3f eigenVectorsPCA;
    Eigen::Vector3f eigenValuesPCA;
    std::vector<int> neibors;
} SNeiborPCA;

/**
 * @struct SClusterFeature
 * @brief 点云聚类目标特征结构体
 *
 * 存储单个聚类簇（障碍物/物体/区域）的所有几何与形态特征，
 * 用于障碍物识别、跟踪、规划、避障。
 * 包含：基础包围盒、PCA主方向、中心点、OBB定向包围盒及类别标签。
 */
typedef struct
{
    // ====================== 基础几何信息 ======================
    int pnum;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    float zmean;

    // ====================== PCA方向信息 ======================
    float d0[3];  // 主方向
    float d1[3];  // 次方向
    float center[3];

    // ====================== 定向包围盒OBB ====================
    float obb[8];

    // ====================== 分类结果 =========================
    int cls;      // 类别
} SClusterFeature;

int FilterGndForPos(float* outPoints,float*inPoints,int inNum);
int CalNomarls(float *nvects, float *fPoints,int pointNum,float fSearchRadius);
int CalGndPos(float *gnd, float *fPoints,int pointNum,float fSearchRadius);
int GetNeiborPCA(SNeiborPCA &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius);
int CorrectPoints(float *fPoints,int pointNum,float *gndPos);

// 地面上物体分割
/**
 * @brief 对地面以上障碍物点云执行背景平面+前景物体两级分割，输出分类标签
 *
 * 执行步骤：
 * 1. 将输入四点式浮点数组点云转换为PCL PointXYZ标准点云结构；
 * 2. 构建KD-Tree用于邻域快速检索；
 * 3. SegBG：邻域半径0.5m生长分割大面积背景平面（墙体、平整立面），背景标记为1；
 * 4. SegObjects：邻域半径0.7m对剩余未标记点聚类分割独立前景物体，前景标记值≥10；
 * 5. FreeSeg做噪声点过滤与标签修正；
 *
 * @param[out] pLabel 输出分类标签数组
 *            1 = 背景大面积平面（墙面/围栏）
 *           ≥10 = 前景独立实体障碍物
 *            0 = 噪声、细碎无效点
 * @param[in] fPoints 输入障碍物点云数组，四点一组[x,y,z,占位]
 * @param[in] pointNum 障碍物点总数量
 * @return int 固定返回0，无失败状态码区分
 *
 * @note 依赖PCL库点云、KDTree模块；内部依次调用SegBG、SegObjects、FreeSeg
 * @warning
 * 1. 每帧新建PCL点云和KDTree，无对象复用，存在一定性能开销
 * 2. 无空指针、点数合法性校验，非法入参易内存崩溃
 * 3. 返回值无异常反馈，无法识别空点云、分割失效等工况
 */
int AbvGndSeg(int *pLabel, float *fPoints, int pointNum);

/**
 * @brief 区域增长分割高大背景平面（墙体、建筑立面、树木等），标记背景标签1
 *
 * 核心算法流程：
 * 1. 标签初始化：Z>4m点预标记为候选背景(1)，其余初始0；仅4m<Z<6m的高点作为区域增长种子点
 * 2. 迭代区域增长：从种子点向外半径邻域扩散
 *    - x<44.8：邻域搜索半径使用传入fSearchRadius
 *    - x≥44.8：放大至1.5倍fSearchRadius扩大搜索范围
 * 3. 邻域内未标记(0)点统一设为背景1；仅Z>0.2m的新标记点加入种子队列继续扩散，过滤贴近地面杂点
 *
 * @param[out] pLabel 点标签数组：1=背景平面，0=未分类
 * @param[in] cloud 待分割障碍物点云
 * @param[in] kdtree 预构建KD-Tree，用于半径邻域搜索
 * @param[in] fSearchRadius 基础区域生长半径
 * @return int 固定返回0，无错误状态反馈
 *
 * @note 设计思路：由高处向下生长，优先捕获建筑、高墙这类大面积直立背景结构
 * @warning
 * 1. 硬编码高度阈值4m、6m、0.2m，距离阈值44.8无宏定义，修改需改动源码
 * 2. 无空指针、点云数量合法性校验
 * 3. 墙体高度不足4m时无法被初始选为种子，低矮墙面极易漏分割
 */
int SegBG(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);
/**
 * @brief 对未标记非背景点做欧式聚类，筛选有效前景障碍物物体，不合格簇重归类为背景/噪声
 *
 * 核心流程：
 * 1. 初始化前景物体起始编号labelId=10；仅处理当前标签pLabel=0的未分类点
 * 2. 筛选种子点：点Z高度>0.4m才启动聚类，低矮杂点跳过
 * 3. FindACluster执行邻域欧式聚类，给整个簇分配当前labelId编号，同时输出簇包围盒、点数量、平均高度等特征cf
 * 4. 根据簇尺寸、高度、点数、距离雷达远近多组硬阈值判定簇是否不合格：
 *    - 长宽任一超15m / 长宽均超10m → 标记isBg=2（巨型平面归背景）
 *    - 长宽任一超6m且簇平均高度<1.5m → isBg=3（低矮长条结构归背景）
 *    - 长宽均小于1.5m但最高点超2.5m → isBg=4（细高杆类异常结构）
 *    - 点数过少（<5 或 <10且离雷达近cx<50）→ isBg=5（噪声小簇）
 *    - 簇平均高度过高>3m或过低<0.3m → isBg=6（高低异常）
 * 5. 不合格簇：批量将簇内点标签改为对应isBg背景值；合格物体保留labelId并自增编号
 *
 * @param[out] pLabel 标签数组：1=背景，≥10=有效前景物体，2/3/4/5/6=各类不合格背景簇
 * @param[in] cloud 障碍物点云
 * @param[in] kdtree 预构建KD-Tree用于半径邻域搜索
 * @param[in] fSearchRadius 聚类欧式邻域半径
 * @return int 最终有效前景物体总个数（labelId-10）
 *
 * @note 依赖FindACluster完成单簇提取与簇特征计算SClusterFeature
 * @note 硬编码判定阈值全部写死无宏：0.4m种子高度、15/10/6/1.5m长宽、2.5/3/0.3m高度、5/10最小点数、50距离门限
 * @warning
 * 1. 无空指针、点云数量安全校验；大量魔法数值不利于场景适配调参
 * 2. 低矮墙面、薄墙体因Z高度不足、簇尺寸偏大极易被判定为背景，爬墙工况匹配特征不足
 */
int SegObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);

SClusterFeature FindACluster(int *pLabel, int seedId, int labelId, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius, float thrHeight);

int CalFreeRegion(float *pFreeDis, float *fPoints,int *pLabel,int pointNum);
int FreeSeg(float *fPoints,int *pLabel,int pointNum);
int CompleteObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);
int ExpandObjects(int *pLabel, float* fPoints, int pointNum, float fSearchRadius);
int ExpandBG(int *pLabel, float* fPoints, int pointNum, float fSearchRadius);

/**
 * @brief 栅格最低点多层高度阈值粗分割地面点，输出0/1标签，返回有效地面点总数
 *
 * 完整筛选流程：
 * 1. 构建XY栅格高度图，遍历所有点记录每个栅格内最小Z高度；
 * 2. 初步筛选：点Z高度 ≤ 所属栅格最低点+0.5m，标记为候选地面(pLabel=1)；
 * 3. 距离分层高度约束修正标签：
 *    - 候选点Z>1m直接取消地面标签；10m半径内Z>0.5m取消标签；
 *    - 非候选点：20m半径内Z<0.2m强制标记为地面；
 * 4. 20m内候选地面点求取平均高度zMean，高于均值+0.4m的近场候选点剔除；
 * 5. 统计最终pLabel=1的地面点总数作为返回值
 *
 * @param[out] pLabel 输出标签数组，长度pointNum；1=地面点，0=非地面点
 * @param[in] fPoints 输入点云浮点数组，四点一组 [x,y,z,unused]
 * @param[in] pointNum 输入点云总数量
 * @param[in] fSearchRadius 预留搜索半径入参，本函数内部未实际使用
 * @return int 最终标记为地面的点总数量gnum
 *
 * @note 依赖全局栅格宏：GND_IMG_NX1/GND_IMG_NY1/GND_IMG_OFFX1/GND_IMG_OFFY1/GND_IMG_DX1/GND_IMG_DY1
 * @note 距离判定使用坐标平方对比，不开根号：10m→x²+y²<225；20m→x²+y²<400
 * @note 仅基于雷达原始坐标系Z高度判断，无地面倾斜/坡度补偿，仅适配平缓平地
 *
 * @warning
 * 1. 使用calloc手动分配堆内存，无空指针校验，内存不足会崩溃；函数内配套free释放
 * 2. 未校验pLabel、fPoints空指针，pointNum≤0会数组越界
 * 3. fSearchRadius形参定义但全程未参与运算，属于冗余入参
 * 4. 无异常错误码，仅返回点数，无法识别分配失败、无有效点等故障工况
 */
int GndSeg(int* pLabel,float *fPoints,int pointNum,float fSearchRadius);

/**
 * @class PCSeg(Short for Point Cloud Segmentation)
 * @brief 点云分割类
 *
 * 该类实现激光雷达点云的完整处理流水线，包括：
 * 1. 点云体素降采样
 * 2. 地面法向量估计与点云姿态校正
 * 3. 地面/非地面分割
 * 4. 地面上区域：背景/前景障碍物分割
 * 5. 障碍物聚类与OBB包围盒生成
 * 6. 环境特征编码输出
 *
 * 主要用于自动驾驶/机器人导航中的环境感知、可通行区域检测、障碍物提取。
 *
 * @note 依赖PCL库进行点云搜索、PCA及协方差计算
 * @note 内部使用全局工具函数完成算法核心逻辑
 */

class PCSeg
{
public:
    /**
     * @brief 构造函数
     * 初始化体素栅格缓存、状态标志、点云缓存指针
     */
    PCSeg();

    /**
     * @brief 析构函数
     * 释放所有动态分配的内存：体素图像、矫正点云等
     */
    ~PCSeg();
    /**
     * @brief 点云分割主处理函数
     * @param pLabel1  输出：每个点的分类标签
     * @param fPoints1 输入：点云数据 x,y,z,intensity
     * @param pointNum 输入：点数量
     * @return 执行状态
     */
    int DoSeg(int *pLabel, float* fPoints1, int pointNum);
    /**
     * @brief 提取障碍物聚类特征
     * @param fPoints  点云
     * @param pLabel   点标签
     * @param pointNum 点数量
     * @return 执行状态
     */
    int GetMainVectors(float*fPoints, int* pLabel, int pointNum);

   /**
     * @brief 编码障碍物特征供上层模块使用
     * @param pFeas 输出：特征数组
     * @return 执行状态
     */
    int EncodeFeatures(float *pFeas);

    /**
     * @brief 边界检测接口
     * 对点云进行环境边界/路沿/障碍物边缘检测
     * @param fPoints1 输入：点云数据 x,y,z,intensity
     * @param pLabel1  输入/输出：边界检测结果标签
     * @param pointNum 输入：点数量
     * @return 执行状态
     */
    int DoBoundaryDetection(float* fPoints1,int *pLabel1,int pointNum);

    /**
     * @brief 交通线检测接口（车道线/导向箭头/停止线等）
     * 从地面点云中提取交通标线特征
     * @param fPoints1 输入：点云数据 x,y,z,intensity
     * @param pLabel1  输入/输出：交通线检测标签
     * @param pointNum 输入：点数量
     * @return 执行状态
     */    
    int DoTrafficLineDet(float *fPoints1,int *pLabel1,int pointNum);

    /**
     * @brief 对点云执行地面姿态旋转变换校正，把倾斜地面校准为Z轴水平基准面并归零地面高度
     *
     * 变换流程：
     * 1. 依据拟合得到的地面平面法向量，调用GetRTMatrix生成3×3旋转矩阵；目标将地面法向对齐标准竖直方向(0,0,1)
     * 2. 利用旋转矩阵计算地面中心点变换后的Z高度，作为整体高度偏移量gndHeight
     * 3. 遍历所有点，使用旋转矩阵对点XYZ坐标做姿态旋转；Z轴额外减去地面偏移高度，校正后地面点理论z≈0
     * 4. 校正后的坐标原地覆盖写入输入点云缓冲区
     *
     * @param[in,out] fPoints 点云浮点数组，存储格式为每4个float一组：[x, y, z, 占位无效值]，结果直接覆写原数据
     * @param[in] pointNum 输入点云总数量
     * @param[in] gndPos 地面平面参数数组，长度固定6
     *        gndPos[0],gndPos[1],gndPos[2]：地面平均法向量(nx, ny, nz)
     *        gndPos[3],gndPos[4],gndPos[5]：地面面片平均中心点(cx, cy, cz)
     * @return int 固定返回0，不区分空指针、点数非法等异常状态
     *
     * @note 依赖接口：GetRTMatrix() 生成对齐地面法向的旋转矩阵
     * @note RTM一维数组存储3×3旋转矩阵，采用行优先排布
     * @warning
     * 1. 会直接篡改原始fPoints数据，若需要保留原始点云，调用前必须提前拷贝备份
     * 2. 未做空指针校验、pointNum数值合法性校验，非法入参将引发内存崩溃
     * 3. 无异常错误码反馈，无法识别校正失败工况
     */
    int CorrectPoints(float *fPoints,int pointNum,float *gndPos);

    float *PrePoints;
    int numPrePoints = 0;
    float gnd_pos[100*6];
    int gnd_vet_len = 0;
    int laneType=0;
    float lanePosition[2] = {0};
    // vars
    unsigned char *pVImg;
    float gndPos[6];        // 地面位姿
    int posFlag;
    // cluster features
    float pVectors[256*3];
    float pCenters[256*3]; // 重心
    int pnum[256];
    int objClass[256];
    float zs[256];
    float pOBBs[256*8];
    float CBBox[256*7]; //（a,x,y,z,l,w,h）
    int clusterNum;
    float *corPoints;
    int corNum;
};

/**
 * @brief 计算点云在XOY平面投影的PCA
 * @param fPoints 输入点云x,y,z,intensity
 * @param pointNum 点数量
 * @return 包含包围盒信息的SClusterFeature
 * @note 这里并不是计算包围盒，不是AABB short for Axis-Aligned Bounding Box，
 *       而是点云投影的2维PCA。
 */
SClusterFeature CalBBox(float *fPoints, int pointNum);

/**
 * @brief 计算点云簇的旋转包围盒OBB（最优方向包围盒）
 * @param fPoints 输入点云 x,y,z,intensity
 * @param pointNum 点数量
 * @return 包含OBB包围盒、中心点、方向向量及类别等完整簇特征
 * @note OBB short for Oriented Bounding Box
 */
SClusterFeature CalOBB(float *fPoints, int pointNum);

#endif
