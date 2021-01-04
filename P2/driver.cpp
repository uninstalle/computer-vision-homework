
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


void threshold_callback(int canny_threshold, void* pGray);
enum class FitEllipseMode { Default, AMS, Direct };
FitEllipseMode Mode = FitEllipseMode::Default;

int canny_threshold = 50;

void modeChange_callback(int mode, void* pGray)
{
	switch (mode)
	{
	case 0:
		Mode = FitEllipseMode::Default;
		break;
	case 1:
		Mode = FitEllipseMode::AMS;
		break;
	case 2:
		Mode = FitEllipseMode::Direct;
		break;
	default:
		break;
	}
	threshold_callback(canny_threshold, pGray);
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Usage: ./driver [image_src]\n";
		return -1;
	}
	Mat src = imread(argv[1]);
	if (src.empty())
	{
		cout << "Cannot open " << argv[1] << "\n" << endl;
		return -1;
	}
	imshow("source", src);

	// to gray
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	blur(gray, gray, { 3,3 });

	// worse than edge detection, don't use it
	// to binary
	//Mat binary;
	//threshold(gray, binary, 0, 255, THRESH_OTSU);

	constexpr int max_thresh = 255;
	createTrackbar("Threshold:", "source", &canny_threshold, max_thresh, threshold_callback, &gray);
	int mode_value = 0;
	createTrackbar("Fit Mode:", "source", &mode_value, 2, modeChange_callback, &gray);

	// NEED QT SUPPORT!
	//createButton("Default", defaultButtonOnClick, nullptr, QT_RADIOBOX, true);
	//createButton("AMS", AMSButtonOnClick, nullptr, QT_RADIOBOX, false);
	//createButton("Direct", directButtonOnClick, nullptr, QT_RADIOBOX, false);

	threshold_callback(50, &gray);
	waitKey();
	return 0;
}

void threshold_callback(int canny_threshold, void* pGray)
{
	Mat& gray = *static_cast<Mat*>(pGray);
	// edge detection
	Mat canny;
	Canny(gray, canny, canny_threshold, canny_threshold * 3);


	imshow("canny", canny);

	// find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	// RETR_CCOMP makes all outer contours be the only parents of their corresponding inner contours
	findContours(canny, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	cout << "Contours found: " << contours.size() << endl;

	// fit ellipses
	vector<RotatedRect> ellipses;
	for (int i = 0; i < contours.size(); ++i)
	{
		// since every outer contour has no parent, check hierarchy[i][2] to select only outer contour
		if (contours[i].size() > 5 && hierarchy[i][2] == -1)
			if (Mode == FitEllipseMode::Default)
				ellipses.push_back(fitEllipse(contours[i]));
			else if (Mode == FitEllipseMode::AMS)
				ellipses.push_back(fitEllipseAMS(contours[i]));
			else
				ellipses.push_back(fitEllipseDirect(contours[i]));
	}
	cout << "Ellipses found: " << ellipses.size() << endl;

	// draw contours
	Mat result = Mat::zeros(gray.rows, gray.cols, CV_8UC3);
	drawContours(result, contours, -1, { 255,255,255 });
	Scalar c;
	if (Mode == FitEllipseMode::Default)
		c = { 0,0,255 };
	else if (Mode == FitEllipseMode::AMS)
		c = { 0,255,0 };
	else
		c = { 255,0,0 };

	for (auto& ellipse : ellipses)
	{
		Point2f pts[4];
		ellipse.points(pts);

		for (int i = 0; i < 4; ++i) {
			line(result, pts[i], pts[(i + 1) % 4], c, 1, LINE_AA);
		}
		cv::ellipse(result, ellipse, c, 1, LINE_AA);
	}

	imshow("contours", result);
}