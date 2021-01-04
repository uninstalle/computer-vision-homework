// Example 18-1. Reading a chessboard¡¯s width and height, reading and collecting
// the requested number of views, and calibrating the camera
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "reader.h"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;


int main(int argc, char* argv[]) {

	if (argc < 4 || argc > 6) {
		cout << "Error: Unexpected number of input parameters\n";
		cout << "Usage: " << argv[0] << " [ImageDirectory] [WidthCornerCount] [HeightCornerCount] [optional:ShowRemapResult] [optional:ShowCornerFound]\n";
		cout << "ImageDirectory: Directory of images, end with \\\n";
		cout << "WidthCornerCount: The number of corners in one row\n";
		cout << "HeightCornerCount: The number of corners in one col\n";
		cout << "ShowRemapResult(default n): y/n, if y, a widget will be opened to show the remap result of all input images\n";
		cout << "ShowCornerFound(default n): y/n, if y, a widget will be opened to show the corners found of all input images\n";
		return -1;
	}
	// 12 * 12

	std::string imageDir(argv[1]);
	int widthCorner = std::stoi(argv[2]),
		heightCorner = std::stoi(argv[3]);

	bool showRemap = false;
	bool showCorners = false;
	if (argc >= 5)
	{
		std::string remap = argv[4];
		if (remap == "y")
			showRemap = true;
	}
	if (argc == 6)
	{
		std::string corner = argv[5];
		if (corner == "y")
			showCorners = true;
	}

	cout << "Loading images...\n";
	vector<cv::Mat> images = loadImages(imageDir);

	// rotate to the same direction
	for (auto& image : images)
	{
		if (image.rows > image.cols)
			cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
	}

	cv::Size cornerSize(widthCorner, heightCorner);

	// Collection of corner position for every image
	vector<vector<cv::Point2f>> imagePoints;
	// Collection of corner coordinates for every image in object coordinate system
	vector<vector<cv::Point3f>> objectPoints;

	cout << "Searching corners...\n";
	for (auto& image : images)
	{
		cv::Mat grayImage;
		cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

		vector<cv::Point2f> corners;
		bool patternWasFound = cv::findChessboardCorners(grayImage, cornerSize, corners);
		if (showCorners)
			drawChessboardCorners(image, cornerSize, corners, patternWasFound);
		imagePoints.push_back(corners);

		if (patternWasFound)
		{
			vector<cv::Point3f> opts;
			int boardsNum = widthCorner * heightCorner;
			opts.resize(boardsNum);
			for (int i = 0; i < boardsNum; i++)
			{
				opts[i] = cv::Point3f(static_cast<float>(i / widthCorner),
					static_cast<float>(i % widthCorner), 0.0f);
			}
			objectPoints.push_back(opts);
		}

		if (showCorners)
		{
			cv::imshow("Calibration", image);
			cv::waitKey();
		}
	}
	if (showCorners)
		cv::destroyWindow("Calibration");
	cout << "Calibrating..." << endl;


	cv::Mat cameraMatrix, distCoeffs;
	double err = cv::calibrateCamera(objectPoints, imagePoints, images[0].size(),
		cameraMatrix, distCoeffs, cv::noArray(), cv::noArray(),
		cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

	// save result
	cout << "Reprojection error is " << err << endl;
	cout << "Storing Intrinsics.xml files...\n";

	cv::FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);
	fs << "ImageWidth" << images[0].cols << "ImageHeight" << images[0].rows
		<< "CameraMatrix" << cameraMatrix << "DistortionCoefficients" << distCoeffs;
	fs.release();

	cout << "Done.\n";

	if (showRemap) {

		cv::Mat map1, map2;
		cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix
			, images[0].size(), CV_16SC2, map1, map2);

		for (auto& image : images)
		{
			cv::remap(image, image, map1, map2, cv::INTER_LINEAR);
			cv::imshow("Undistorted", image);
			cv::waitKey();
		}
	}

	return 0;
}