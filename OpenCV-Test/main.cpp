
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
	Mat frame;												// OpenCV에서 가장 기본이 되는 Matrix 구조체(이미지를 읽어 해당 정보를 Mat형태로 변환)
	VideoCapture cap;										// 동영상 불러오기
	cap.open(0);											// 동영상 열기(Camera 열기) + 카메라번호(0(내장 우선))
	if (!cap.isOpened())
	{
		cout << "Error! Cannot open the camera" << endl;
		return -1;
	}
	while (1)
	{
		cap.read(frame);									// 비디오의 한 프레임씩 read
		imshow("LIVE", frame);								// 프레임을 화면에 display
		if (waitKey(5) >= 0)								// 5만큼 키입력을 대기하고, 발생시 반환
			break;
	}
	return 0;
}

/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argv, char** argc)
{
	Mat frame;
	VideoCapture cap;

	//int deviceID = 0;
	//int apiID = cv::CAP_ANY;
	//cap.open(deviceID + apiID);

	cap.open(0);

	if (!cap.isOpened())
	{
		cerr << "Unable to open Camera\n";
		return -1;
	}

	while (1)
	{
		cap.read(frame);

		imshow("LIVE", frame);
		if (waitKey(1) >= 0)
			break;
	}
	return 0;
}
*/

/*
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	int size_x = 5;
	int size_y = 5;
	Mat kernel1 = getStructuringElement( MORPH_RECT, Size(size_x, size_y), Point(int(size_x / 2), int(size_y / 2)) );
	Mat kernel2 = getStructuringElement(MORPH_CROSS, Size(size_x, size_y), Point(int(size_x / 2), int(size_y / 2)));
	Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(size_x, size_y), Point(int(size_x / 2), int(size_y / 2)));

	std::string windowTitle = "Origin";
	Mat i1 = imread("D:\\test\\copy.jpg", IMREAD_GRAYSCALE);
	Mat i2, i3, i4;
	Mat i21, i31, i41;
	
	//erode(i1, i2, cv::Mat(), cv::Point(-1, -1), 1);
	erode(i1, i2, kernel1, cv::Point(int(size_x / 2), int(size_y / 2)), 3);
	erode(i1, i3, kernel2, cv::Point(int(size_x / 2), int(size_y / 2)), 3);
	erode(i1, i4, kernel3, cv::Point(int(size_x / 2), int(size_y / 2)), 3);

	dilate(i2, i21, kernel1, cv::Point(int(size_x / 2), int(size_y / 2)), 3);
	dilate(i3, i31, kernel2, cv::Point(int(size_x / 2), int(size_y / 2)), 3);
	dilate(i4, i41, kernel3, cv::Point(int(size_x / 2), int(size_y / 2)), 3);

	imshow(windowTitle, i1);
	waitKey();

	imshow("Erosion1", i2);
	waitKey();
	imshow("Erosion2", i3);
	waitKey();
	imshow("Erosion3", i4);
	waitKey();

	imshow("Dilation1", i21);
	waitKey();
	imshow("Dilation2", i31);
	waitKey();
	imshow("Dilation3", i41);
	waitKey();


	return 0;
}
*/

/*
// FLANN(Fast Library for Approximate Nearest Neighbors) in OpenCV
// Create a kdtree for searching the data.
cv::flann::KDTreeIndexParams index_params;
cv::flann::Index kdtree(data, index_params);
//...
// Search the nearest vector to some query
int k = 1;
Mat nearest_vector_idx(1, k, DataType <int >::type);
Mat nearest_vector_dist(1, k, DataType <float >::type);
kdtree.knnSearch(query, nearest_vector_idx, nearest_vector_dist, k);
*/

/*
// SIFT(Scale Invariant Feature Transform) Extraction With OpenCV
// Detect key points.
auto detector = SiftFeatureDetector::create();
vector <cv::KeyPoint > keypoints;
detector->detect(input, keypoints);

// Show the keypoints on the image.
Mat image_with_keypoints;
drawKeypoints(input, keypoints, image_with_keypoints);

// extract the SIFT descriptors
auto extractor = SiftDescriptorExtractor::create();
extractor->compute(input, keypoints, descriptors);
*/

/*
#include <opencv2/opencv.hpp >

int main() 
{
	cv::Mat image = cv::Mat::zeros(800, 600, CV_8UC3);
	std::string window_name = "Window name";
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::imshow(window_name, image);
	cv::waitKey();
	for (int r = 0; r < image.rows; ++r) 
	{
		for (int c = 0; c < image.cols; ++c) 
		{
			// WARNING! WRONG TYPE USED!
			//image.at <float >(r, c) = 1.0f;	
			image.at <cv::Vec3b >(r, c) = cv::Vec3b(0,0,128);
		}
		
	}
	cv::imshow(window_name, image);
	cv::waitKey();
	return 0;	
}
*/

/*
#include <opencv2/opencv.hpp >
#include <iostream >
using namespace cv;

int main() 
{
	Mat mat = Mat::zeros(10, 10, CV_8UC3);
	std::cout << mat.at <Vec3b >(5, 5) << std::endl;
	Mat_ <Vec3f > matf3 = Mat_ <Vec3f >::zeros(10, 10);
	std::cout << matf3.at <Vec3f >(5, 5) << std::endl;
	
	getchar();
	return 0;
}
*/

/*
// clang -format off
#include <opencv2/opencv.hpp >
int main() 
{
	cv::Mat image = cv::imread("D:\\test\\lenna.bmp", cv::IMREAD_COLOR);
	std::string window_name = "Window name";
	// Create a window.
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::imshow(window_name, image); // Show image.
	cv::waitKey(); // Don't close window instantly.
	return 0;	
}
*/

//#include <iostream >
//#include <string >
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/opencv.hpp >
// /* 속성 환경변수 : OPENCV_IO_ENABLE_OPENEXR=1 */
//int main(int argc, char** argv)
//{
//	/*
//	cv::Mat m(cv::Size(2048, 1024), CV_32FC1);
//	for (int y = 0; y < m.rows; y++)
//	{
//		for (int x = 0; x < m.cols; x++)
//		{
//			m.at<float>(y, x) = y + x*1e-3;
//		}
//	}
//
//	cv::Rect roi(498, 64, 8, 4);
//	std::cout << "test ROI: " << roi << std::endl;
//	std::cout << "source: " << std::endl;
//	std::cout << m.type() << ": " << m.depth() << " x " << m.channels() << " " << m.size() << std::endl;
//	std::cout << m(roi) << std::endl;
//
//	{
//		cv::imwrite("D:\\test\\test.exr", m);
//		cv::Mat m2 = cv::imread("D:\\test\\test.exr");
//		std::cout << "imread (default): " << std::endl;
//		std::cout << m2.type() << ": " << m2.depth() << " x " << m2.channels() << " " << m2.size() << std::endl;
//		std::cout << m2(roi) << std::endl;
//	}
//	*/
//	
//	using Matf = cv::Mat_ <float >;
//	Matf image = Matf::zeros(10, 10);
//	image.at <float >(5, 5) = 42.42f;
//
//	std::string f = "D:\\test\\test.exr";
//	cv::imwrite(f, image);
//	Matf copy = cv::imread(f, cv::IMREAD_UNCHANGED);
//	std::cout << copy.at <float >(5, 5) << std::endl;
//	
//	getchar();
//	return 0;
//}

/*
#include <opencv2/core.hpp >
#include <opencv2/highgui.hpp >

int main(int argc, char** argv)
{
	cv::Mat image = cv::imread("D:\\test\\lenna.bmp", cv::IMREAD_GRAYSCALE);
	cv::imwrite("D:\\test\\copy.jpg", image);
	return 0;
}
*/

//#include <opencv2/imgcodecs.hpp >
/*
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat i1 = imread("D:\\test\\lenna.bmp", IMREAD_GRAYSCALE);
	Mat_ <uint8_t > i2 = imread("D:\\test\\lenna.bmp", IMREAD_GRAYSCALE);
	cout << (i1.type() == i2.type()) << endl;

	//cout << "OPENCV VERSION : " << CV_VERSION << endl;
	getchar();
	return 0;
}
*/