#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

using namespace cv;

class VideoMaker
{
private:
	std::string dir;
	VideoCapture videoCapture;
	std::vector<Mat> images;
	const String subtitle = "ID";
	Size videoSize;

	void loadVideo();
	void loadImages();
	Size getNewVideoSize() const;
	void addSubtitle(Mat& mat, Size imageSize);
	void writeImageFrame(VideoWriter& vWriter, Size frameSize, double frameRate);
	void writeVideoFrame(VideoWriter& vWriter, Size frameSize);
public:
	VideoMaker() = default;
	VideoMaker(std::string dir) :dir(std::move(dir)) { loadAssets(); }
	VideoMaker(std::string dir, int w, int h) :dir(std::move(dir)), videoSize(Size{ w,h }) { loadAssets(); }

	void loadAssets();
	void writeNewVideo();
};

