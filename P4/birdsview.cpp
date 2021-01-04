#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;

int main(int argc, char* argv[]) {

	if (argc != 5) {
		cout << "Error: Unexpected number of input parameters\n";
		cout << "Usage: " << argv[0] << " [ImagePath] [WidthCornerCount] [HeightCornerCount] [CalibrationResult]\n";
		cout << "ImagePath: The path of the source image\n";
		cout << "WidthCornerCount: The number of corners in one row\n";
		cout << "HeightCornerCount: The number of corners in one col\n";
		cout << "CalibrationResult: The path of the result xml file of calibration\n";
		cout << "Press 's' for lower birdseye view, and 'w' for higher, Esc to exit.\n";
		return -1;
	}

	std::string imagePath(argv[1]);
	int widthCorner = std::stoi(argv[2]),
		heightCorner = std::stoi(argv[3]);
	std::string resPath(argv[4]);

	cv::FileStorage fs(resPath, cv::FileStorage::READ);
	cv::Mat intrinsic, distortion;

	fs["CameraMatrix"] >> intrinsic;
	fs["DistortionCoefficients"] >> distortion;

	if (!fs.isOpened() || intrinsic.empty() || distortion.empty()) {
		cout << "Error: Couldn't load intrinsic parameters from " << resPath << endl;
		return -1;
	}
	fs.release();

	cv::Mat imageSrc = cv::imread(imagePath);
	if (imageSrc.empty()) {
		cout << "Error: Couldn't load image " << imagePath << endl;
		return -1;
	}

	// remap
	cv::Mat image;
	cv::undistort(imageSrc, image, intrinsic, distortion, intrinsic);
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);

	cv::Size cornerSize(widthCorner, heightCorner);

	vector<cv::Point2f> corners;
	bool found = cv::findChessboardCorners(gray, cornerSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
	if (!found) {
		cout << "Error: Couldn't acquire checkerboard on " << imagePath << ", only found "
			<< corners.size() << " of " << widthCorner * heightCorner << " corners\n";
		return -1;
	}

	// Get Subpixel accuracy on those corners
	cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
		cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));

	cv::Point2f imgPts[4];

	// get the coordinates of four corner points of the corners on the checkerboard
	imgPts[0] = corners[0];
	imgPts[1] = corners[widthCorner - 1];
	imgPts[2] = corners[(heightCorner - 1) * widthCorner];
	imgPts[3] = corners[(heightCorner - 1) * widthCorner + widthCorner - 1];

	// draw corner points in order: B,G,R,YELLOW
	cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
	cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
	cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
	cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);

	// draw corners
	cv::drawChessboardCorners(image, cornerSize, corners, found);
	cv::imshow("Checkers", image);


	cv::Point2f objPts[4];
	cv::Mat homography;

	cout << "\nPress 's' for lower birdseye view, and 'w' for higher, Esc to exit" << endl;
	double projectionSize = 15;
	cv::Mat birdseyeImage;
	while (true) {

		// left top
		objPts[0].x = 100;
		objPts[0].y = 100;
		// right top
		objPts[1].x = 100 + (widthCorner - 1) * projectionSize;
		objPts[1].y = 100;
		// left bottom
		objPts[2].x = 100;
		objPts[2].y = 100 + (heightCorner - 1) * projectionSize;
		// right bottom
		objPts[3].x = 100 + (widthCorner - 1) * projectionSize;
		objPts[3].y = 100 + (heightCorner - 1) * projectionSize;

		// get the homography mat from object coordinate system to image coordinate system
		homography = cv::getPerspectiveTransform(objPts, imgPts);

		cv::warpPerspective(image, birdseyeImage, homography, image.size(),
			cv::WARP_INVERSE_MAP | cv::INTER_LINEAR, cv::BORDER_CONSTANT,
			cv::Scalar::all(0));
		cv::imshow("BirdsEye", birdseyeImage);
		int key = cv::waitKey() & 255;
		if (key == 'w')
			projectionSize += 0.5;
		if (key == 's')
			projectionSize -= 0.5;
		if (key == 27)
			break;
	}

	// SHOW ROTATION AND TRANSLATION VECTORS
	//
	vector<cv::Point2f> image_points;
	vector<cv::Point3f> object_points;
	for (int i = 0; i < 4; ++i) {
		image_points.push_back(imgPts[i]);
		object_points.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
	}
	cv::Mat rvec, tvec, rmat;
	cv::solvePnP(object_points, 	// 3-d points in object coordinate
		image_points,  	// 2-d points in image coordinates
		intrinsic,     	// Our camera matrix
		cv::Mat(),     	// Since we corrected distortion in the
						 // beginning,now we have zero distortion
						 // coefficients
		rvec, 			// Output rotation *vector*.
		tvec  			// Output translation vector.
	);
	cv::Rodrigues(rvec, rmat);

	// PRINT AND EXIT
	cout << "rotation matrix: " << rmat << endl;
	cout << "translation vector: " << tvec << endl;
	cout << "homography matrix: " << homography << endl;
	cout << "inverted homography matrix: " << homography.inv() << endl;

	return 0;
}
