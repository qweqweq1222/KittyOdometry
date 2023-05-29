#pragma once
#include <iostream>
#include <cmath>
#include <map>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include <experimental/filesystem>
#include <string>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

const int MAX_FEATURES = 200;
const int MIN_FEATURES = 10;
const float GOOD_MATCH_PERCENT = 1.0f;
const float DEPTH_TRASH = 100.0f;
const int SAME_POINTS = 30;
const int NUM_OF_FRAMES = 1500;
const int LOCAL_TRASH = 2;
const int START_KEY_FRAME = 0;

struct SnavelyReprojectionErrorPtsOld 
{
	SnavelyReprojectionErrorPtsOld(double observed_x, double observed_y, double fx, double fy, double cx, double cy,vector<double> camera_params)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), camera_params(camera_params) {}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const z_coord, T* residuals) const 
	{


		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];

		T x = z_coord[2] * (T(observed_x) - T(camera_params[0])) / T(camera_params[2]);
		T y = z_coord[2] * (T(observed_y) - T(camera_params[1])) / T(camera_params[3]);
	
		P3[0] = T(cos(b) * cos(g)) * x - T(sin(g) * cos(b)) * y + T(sin(b)) * z_coord[2] + alpha_t[3];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * x + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * y - T(sin(a) * cos(b)) * z_coord[2] + alpha_t[4];
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * x + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * y + T(cos(a) * cos(b)) * z_coord[2] + alpha_t[5];
		
		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x)) / T(100);
		residuals[1] = (predicted_y - T(observed_y)) / T(100);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const vector<double> camera_params) 
	{
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorPtsOld, 2, 6, 3>(
			new SnavelyReprojectionErrorPtsOld(observed_x, observed_y, fx, fy, cx, cy, camera_params)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
	vector<double> camera_params;
};



struct SnavelyReprojectionErrorPtsOldQUAT 
{
	SnavelyReprojectionErrorPtsOldQUAT(double observed_x, double observed_y, double fx, double fy, double cx, double cy) : 
		observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const quat_t, const T* const z, T* residuals) const
	{

		T P3[3];

		T theta = quat_t[0];
		T xv = quat_t[1];
		T yv = quat_t[2];
		T zv = quat_t[3];

		T tx = quat_t[4];
		T ty = quat_t[5];
		T tz = quat_t[6];

		T x = z[0] * (T(observed_x) - T(cx)) / T(fx);
		T y = z[0] * (T(observed_y) - T(cy)) / T(fy);

		T one = T(1);
		P3[0] = (cos(theta) + ( one - cos(theta)) * xv * xv) * x + ((one - cos(theta)) * xv * yv - sin(theta) * zv) * y + ((one - cos(theta)) * xv * zv + sin(theta) * yv) * z[0] + tx;
		P3[1] = ((one - cos(theta)) * yv * xv + sin(theta) * zv) * x + (cos(theta) + (one - (cos(theta))) * yv * yv) * y + ((one - (cos(theta))) * yv * zv - (sin(theta)) * xv) * z[0] + ty;
		P3[2] = ((one - cos(theta)) * zv * xv - sin(theta) * yv) * x + ((one - cos(theta)) * zv * yv + sin(theta) * xv) * y + (cos(theta) + (one - cos(theta)) * zv * zv) * z[0] + tz;

		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x));
		residuals[1] = (predicted_y - T(observed_y));
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double fx, const double fy, const double cx, const double cy)
	{
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorPtsOldQUAT, 2, 7, 1>(new SnavelyReprojectionErrorPtsOldQUAT(observed_x, observed_y, fx, fy, cx, cy)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
};

struct SnavelyReprojectionErrorPts {
	SnavelyReprojectionErrorPts(double observed_x, double observed_y, double fx, double fy, double cx, double cy, bool is_ground)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), is_ground(is_ground){}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const pt3d, T* residuals) const {


		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];

		P3[0] = T(cos(b) * cos(g)) * (pt3d[0]) - T(sin(g) * cos(b)) * (pt3d[1]) + T(sin(b)) * (pt3d[2]) + alpha_t[3];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * (pt3d[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * (pt3d[1]) - T(sin(a) * cos(b)) * (pt3d[2]) + alpha_t[4];
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * (pt3d[0]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * (pt3d[1]) + T(cos(a) * cos(b)) * (pt3d[2]) + alpha_t[5];

		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x));
		residuals[1] = (predicted_y - T(observed_y));
		residuals[2] = is_ground == true ? (pt3d[1] - T(1.65)) : T(0);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const bool is_ground) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorPts, 3, 6, 3>(
			new SnavelyReprojectionErrorPts(observed_x, observed_y, fx, fy, cx, cy, is_ground)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
	bool is_ground;
};

struct SnavelyReprojectionErrorPtsZ {
	SnavelyReprojectionErrorPtsZ(double observed_x, double observed_y, double fx, double fy, double cx, double cy, bool is_ground)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), is_ground(is_ground) {}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const z, T* residuals) const {


		T P3[3];
		T b = alpha_t[0];
		
		T x = z[0] * (observed_x - cx) / fx;
		T y = z[0] * (observed_y - cy) / fy;

		P3[0] = T(cos(b)) * x  + T(sin(b)) * z[0] + alpha_t[1];
		P3[1] = y;
		P3[2] = T(-sin(b)) * x + T(cos(b)) * z[0] + alpha_t[2];

		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x));
		residuals[1] = (predicted_y - T(observed_y));
		residuals[2] = is_ground == true ? (y - T(1.65)) : T(0);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const bool is_ground) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorPtsZ, 3, 3, 1>(
			new SnavelyReprojectionErrorPtsZ(observed_x, observed_y, fx, fy, cx, cy, is_ground)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
	bool is_ground;
};



struct SnavelyReprojectionError_ {
	SnavelyReprojectionError_(double observed_x, double observed_y, double fx, double fy, double cx, double cy, Vec3f pt, double y)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), pt(pt), y(y) {}

	template <typename T>
	bool operator()(const T* const alpha_t, T* residuals) const {


		T P3[3];
		T a = T(0);
		T b = alpha_t[0];
		T g = T(0);
		// Ceres не позволяет умножать Opencv матрицы на объекты типа T и приводить вообще T к чему-то, поэтому руками выполняем R * X + t
		P3[0] = T(cos(b) * cos(g)) * T(pt[0]) - T(sin(g) * cos(b)) * T(pt[1]) + T(sin(b)) * T(pt[2]) + alpha_t[1];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * T(pt[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * T(pt[1]) - T(sin(a) * cos(b)) * T(pt[2]) + T(y); // косяк 
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * T(pt[0]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * T(pt[1]) + T(cos(a) * cos(b)) * T(pt[2]) + alpha_t[2];
		
		P3[0] = T(cos(b) * cos(g)) * T(pt[0]) + T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * T(pt[1]) + T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * T(pt[2]);
		P3[1] = T(-sin(g) * cos(b)) * T(pt[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * T(pt[1]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * T(pt[2]);
		P3[2] = T(sin(b)) * T(pt[0]) + T(-sin(a) * cos(b)) * T(pt[1]) + T(cos(a) * cos(b)) * T(pt[2]);
		
		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x)) / T(50);
		residuals[1] = (predicted_y - T(observed_y)) / T(50);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const Vec3f pt, const double y) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_, 2, 3>(
			new SnavelyReprojectionError_(observed_x, observed_y, fx, fy, cx, cy, pt, y)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
	double y;
	Vec3f pt;
};

struct KeyPointMatches
{
	vector<DMatch> matches;
	vector<KeyPoint> kp1, kp2;

	KeyPointMatches(vector<DMatch> matches_, vector<KeyPoint> kp1_,
		vector<KeyPoint> kp2_) :matches(matches_), kp1(kp1_), kp2(kp2_) {};

	KeyPointMatches() = default;
	~KeyPointMatches() = default;
};
struct CameraInfo
{
	Mat cameraMatrix, rotMatrix, transVector;
	CameraInfo(Mat camera_matrix, Mat rot_matrix, Mat trans_vector) {
		camera_matrix.convertTo(camera_matrix, CV_32F, 1.0);
		rot_matrix.convertTo(rot_matrix, CV_32F, 1.0);
		trans_vector.convertTo(trans_vector, CV_32F, 1.0);

		cameraMatrix = camera_matrix;
		rotMatrix = rot_matrix;
		transVector = trans_vector;
	};
	~CameraInfo() = default;

};
struct ForOptimize
{
	Mat R, t;
	vector<Point3f> pts_3d;
	vector<Point2f> pts_2d;

	ForOptimize(Mat R_, Mat t_, vector<Point3f>& pts3d, vector<Point2f>& pts2d)
	{
		R_.convertTo(R_, CV_32F, 1.0);
		t_.convertTo(t_, CV_32F, 1.0);
		R = R_;
		t = t_;
		pts_3d = pts3d;
		pts_2d = pts2d;
	}
	~ForOptimize() = default;
};

KeyPointMatches AlignImages(Mat& im1, Mat& im2); // find features
CameraInfo Decompose(Mat proj_matrix); // decompose given P matrix of a camera

Mat CalculateDisparity(const cv::Mat& left_image, const cv::Mat& right_image);
Mat TAP(const Mat& R, const Mat& t);
Mat T(Mat& R, Mat& t);

pair<Mat, Mat> EstimateMotion(Mat left, Mat right, Mat next, Mat P_left, Mat P_right);
pair<Mat, Mat> EstimateNoDynamicMotion(Mat left, Mat right, Mat next, Mat left_segment, Mat P_left, Mat P_right, std::vector<int> dynamic);
std::vector<std::vector<KeyPoint>> GetSamePoints(fs::directory_iterator left, fs::directory_iterator next, const int& N_features);
std::vector<double> Transform_vec(const Mat answer);

Mat Recover(double* pt);

void VisualOdometry(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight, const int step);
void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight);
void SimplifiedOdometry(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight,
	std::fstream& myfile_);
void VisualNoDynamic(const std::string& left_path, const std::string& left_path_segment, const std::string& right_path, const std::string& input,
	const Mat& PLeft, const Mat& PRight, std::vector<int> dynamic, const int step);
void OdometryALL(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight, const int step);
void OdometryAXZ(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, 
	const Mat& PRight, std::vector<int>& dynamic, const std::string& segment, const std::string& y_coord);
void OdometryOLD(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight,
	std::vector<int>& dynamic, const std::string& segment, const int step);
void OdometryOLDZ(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight,
	std::vector<int>& dynamic, const std::string& segment, const int step);
void OdometryQUAT(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight, const int step);
Mat ReconstructFromQUAT(double* initial);
vector<double> GetAnglesVecsFromQuat(Mat& R, Mat& t);
