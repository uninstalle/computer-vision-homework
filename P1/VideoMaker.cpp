#include "VideoMaker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>
#include <iostream>
#include <io.h>

void VideoMaker::loadAssets()
{
	loadVideo();
	loadImages();
}

void VideoMaker::loadVideo()
{
	bool isVideoFound = false;
	_finddata_t findData;
	auto hFile = _findfirst((dir + "\\*.avi").c_str(), &findData);
	do
	{
		if (strcmp(findData.name, "output.avi") != 0)
		{
			isVideoFound = true;
			break;
		}
	} while (_findnext(hFile, &findData) != -1);
	if (!isVideoFound)
	{
		std::cout << "Cannot find video " + dir + "\\*.avi\n";
		return;
	}
	String videoPath = dir + "\\" + findData.name;

	videoCapture.open(videoPath);
	if (!videoCapture.isOpened())
	{
		std::cout << "Cannot open video " + videoPath;
	}

	_findclose(hFile);
}

void VideoMaker::loadImages()
{
	_finddata_t findData;
	auto hFile = _findfirst((dir + "\\*.jpg").c_str(), &findData);

	do
	{
		String imagePath = dir + "\\" + findData.name;
		Mat image = imread(imagePath);
		if (image.empty())
		{
			std::cout << "Cannot open image " + imagePath;
			continue;
		}
		images.push_back(image);
	} while (_findnext(hFile, &findData) == 0);

	_findclose(hFile);

	if (images.empty())
	{
		std::cout << "Cannot find image " + dir + "\\*.jpg\n";
	}
}



Size VideoMaker::getNewVideoSize() const
{
	int	minWidth = videoCapture.get(CAP_PROP_FRAME_WIDTH),
		minHeight = videoCapture.get(CAP_PROP_FRAME_HEIGHT);

	for (Mat image : images)
	{
		minWidth += image.cols;
		minHeight += image.rows;
	}

	int count = 1 + images.size();

	return { minWidth / count,minHeight / count };
}

void VideoMaker::writeNewVideo()
{
	VideoWriter videoWriter;
	double frameRate = videoCapture.get(CAP_PROP_FPS);
	if (videoSize == Size(0, 0))
		videoSize = getNewVideoSize();

	videoWriter.open(dir + "\\output.avi", VideoWriter::fourcc('M', 'P', 'E', 'G'), frameRate, videoSize);
	if (!videoWriter.isOpened())
	{
		std::cout << "Open VideoWriter failed " + dir + "\\output.avi \n";
		return;
	}

	writeImageFrame(videoWriter, videoSize, frameRate);

	writeVideoFrame(videoWriter, videoSize);

	videoWriter.release();
}

void VideoMaker::addSubtitle(Mat& mat, Size imageSize)
{
	int fontHeight = 20;
	int baseLine;
	auto ft2 = freetype::createFreeType2();
	try {
		ft2->loadFontData("syst.otf", 0);
	}
	catch (std::exception& e)
	{
		Size textSize = getTextSize(subtitle, HersheyFonts::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
		Point textPos = { imageSize.width / 2 - textSize.width / 2, imageSize.height - textSize.height };
		putText(mat, subtitle, textPos, HersheyFonts::FONT_HERSHEY_SIMPLEX, 1, { 255,255,255 });
		return;
	}
	Size textSize = ft2->getTextSize(subtitle, fontHeight, -1, &baseLine);
	Point textPos = { imageSize.width / 2 - textSize.width / 2, imageSize.height - textSize.height };
	ft2->putText(mat, subtitle, textPos, fontHeight, { 255,255,255 }, -1, LINE_AA, true);
}

void VideoMaker::writeImageFrame(VideoWriter& vWriter, Size frameSize, double frameRate)
{
	Mat frame;
	Mat iFrame;
	for (Mat& image : images)
	{
		resize(image, frame, frameSize);
		addSubtitle(frame, frameSize);
		for (int f = 0; f < frameRate; ++f)
		{
			frame.convertTo(iFrame, -1, f / frameRate);
			vWriter << iFrame;
		}
		for (int f = frameRate - 1; f >= 0; --f)
		{
			frame.convertTo(iFrame, -1, f / frameRate);
			vWriter << iFrame;
		}
	}
}

void VideoMaker::writeVideoFrame(VideoWriter& vWriter, Size frameSize)
{

	Mat frame, vFrame;

	while (videoCapture.read(vFrame))
	{
		resize(vFrame, frame, frameSize);
		addSubtitle(frame, frameSize);
		vWriter << frame;
	}
}
