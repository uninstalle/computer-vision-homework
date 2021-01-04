#pragma once
#include <iostream>
#include <vector>
#include <io.h>
#include <opencv2/highgui.hpp>

// load all images in given directory
std::vector<cv::Mat> loadImages(const std::string& dir)
{
	std::vector<cv::Mat> images;

	// windows io function
	_finddata_t findData;
	auto hFile = _findfirst((dir + "*.jpg").c_str(), &findData);

	do {
		std::string imagePath = dir + findData.name;
		cv::Mat image = cv::imread(imagePath);

		// cv::imread doesn't throws exception, but the return mat is empty
		if (image.empty())
		{
			std::cout << "Cannot read image " + imagePath << std::endl;
			continue;
		}
		images.push_back(image);
	} while (_findnext(hFile, &findData) == 0);

	_findclose(hFile);

	if (images.empty())
	{
		std::cout << "Cannot find any image in " + dir << std::endl;
	}

	return images;
}