#include "lib.h"

const int STEP = 5;


int main(void) {

	float P0[] = { 7.18856e+02, 0.0e+00, 6.071928e+02, 0.00e+00, 0.0e+00, 7.18856e+02, 1.852157e+02, 0.0e+00, 0.00e+00, 0.0e+00, 1.000e+00, 0.0000e+00 };
	float P1[] = { 7.18856e+02, 0.0e+00, 6.071928e+02, -3.861448e+02, 0.0e+00, 7.18856e+02, 1.852157e+02, 0.0e+00, 0.00e+00, 0.0e+00, 1.000e+00, 0.0000e+00 };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	std::string folder_left = "D:/kitti_experiment/lil_dataset/00/image_0/";
	std::string folder_right = "D:/kitti_experiment/lil_dataset/00/image_1/";
	std::string segmented_left = "D:/kitti_experiment/lil_dataset/00/kitti_masks/";

	std::string output  = "D:/kitti_experiment/experiments_results/opt_step_Vquat.txt";
	std::string output_vanilla = "D:/kitti_experiment/experiments_results/opt_step_Vanilla.txt";
	std::vector dynamic_classes = { 11,12,13,14,15,16,17,18 };

	//OdometryOLD(folder_left, folder_right, output, P_left, P_right, dynamic_classes, segmented_left, 3);
	//OdometryQUAT(folder_left, folder_right, output, P_left, P_right, 5);
	VisualOdometry(folder_left, folder_right, output_vanilla, P_left, P_right, 5);
	return 0;
}


/*
void check_new_hypothes(const Mat& PLeft, const Mat& PRight, const string& left_camera, const string& right_camera, const int step, const int number_of_frames)
{
	fs::directory_iterator left_iterator(left_camera);
	fs::directory_iterator right_iterator(right_camera);
	fs::directory_iterator next_iterator(left_camera);
	::advance(left_iterator, START_KEY_FRAME); // если хотим начать не с 0-го кадра 
	::advance(right_iterator, START_KEY_FRAME);
	::advance(next_iterator, START_KEY_FRAME + step);

	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);
	Mat dst_norm, dst_norm_scaled;
	for (int i = 0; i < number_of_frames; ++i)
	{
		Mat left = imread((*left_iterator).path().u8string());
		Mat right = imread((*right_iterator).path().u8string());
		Mat next = imread((*next_iterator).path().u8string());
		
		KeyPointMatches kpm = AlignImages(left, next);
		for (auto& k : kpm.matches)
			line(left, kpm.kp1.at(k.queryIdx).pt, kpm.kp2.at(k.trainIdx).pt, Scalar(255, 255, 255), 3);

		imshow("flow", left);
		waitKey(0);

		waitKey(0);
		::advance(left_iterator, step); // если хотим начать не с 0-го кадра 
		::advance(right_iterator, step);
		::advance(next_iterator, step);
	}
}
int main(void) {
	float P0[] = { 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00, 
				   0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
		           0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
	float P1[] = { 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, - 3.798145000000e+02,
		           0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00, 
		           0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	std::string folder_left = "D:/Kitti/dataset/sequences/12/image_0/";
	std::string folder_right = "D:/Kitti/dataset/sequences/12/image_1/";
	check_new_hypothes(P_left, P_right, folder_left, folder_right, 1, 100);
}*/
