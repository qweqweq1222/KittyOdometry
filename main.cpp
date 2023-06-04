#include "lib.h"

const int STEP = 5;


int main(void) {

	/*
	float P0[] = {7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00,
				   0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
		           0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
	float P1[] = { 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, -3.798145000000e+02,
				   0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00, 
		           0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
				   
				   */
	float P0[] = {7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
		           0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
				   0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
	float P1[] = { 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
				   0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
				   0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00 };
	
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	std::string folder_left = "D:/Kitti/dataset/sequences/00/image_0/";
	std::string folder_right = "D:/Kitti/dataset/sequences/00/image_1/";
	std::string segmented_left = "D:/results_v_diplom/kitty00/content/segmentation_masks/";
	//std::string segmented_left = "D:/Kitti/content/big_dataset/content/dataset/";

	std::string output_vanilla = "D:/results_v_diplom/kitty00/Proper/vanilla.txt";
	std::string output_masks  = "D:/results_v_diplom/kitty00/Proper/vanilla_masks.txt";
	std::string output_filter = "D:/results_v_diplom/kitty00/Proper/vanilla_filter.txt";
	std::string output_opt = "D:/results_v_diplom/kitty00/Proper/vanilla_opt.txt";
	std::string output_regul = "D:/results_v_diplom/kitty00/Proper/vanilla_opt_regul.txt";
	std::vector<int> dynamic_classes = { 11,12,13,14,15,16,17,18 };

	//OdometryOLD(folder_left, folder_right, output, P_left, P_right, dynamic_classes, segmented_left, 3);
	//OdometryQUAT(folder_left, folder_right, output, P_left, P_right, 5);
	//VisualOdometry(folder_left, folder_right, output_vanilla, P_left, P_right, 5);
	/*Mat src0 = imread("D:/Рабочий стол/Диплом эксперименты/kitty00/content/segmentation_masks/000000.png");
	cv::cvtColor(src0, src0, CV_BGR2GRAY);
	for (int x = 0; x < src0.cols; ++x)
		for (int y = 0; y < src0.rows; ++y)
			if (int(src0.at<uchar>(y, x)) == 3)
				src0.at<uchar>(y, x) = 255;

	cvtColor(src0, src0, CV_GRAY2RGB);
	line(src0, Point(0, 0), Point(255, 255), Scalar(255, 0, 0), 3);
	imshow("frame", src0);
	waitKey(0);*/


	//VisualOdometry(folder_left, folder_right, output_vanilla, P_left, P_right, 3);
	//VisualNoDynamic(folder_left, segmented_left, folder_right, output_masks, P_left, P_right, dynamic_classes, 3, true);
	//VisualNoDynamic(folder_left, segmented_left, folder_right, output_filter, P_left, P_right, dynamic_classes, 3, true);
	OdometryQUAT(folder_left, segmented_left, folder_right, output_opt, P_left, P_right, 3, dynamic_classes, false);
	OdometryQUAT(folder_left, segmented_left, folder_right, output_regul, P_left, P_right, 3, dynamic_classes, true);
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
