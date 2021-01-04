#pragma once
#include <opencv2/imgproc.hpp>

// Eye Position structure
struct EyePos
{
	int LeftX, LeftY;
	int RightX, RightY;
	bool operator==(const EyePos& ep) const
	{
		return LeftX == ep.LeftX && LeftY == ep.LeftY && RightX == ep.RightX && RightY == ep.RightY;
	}
};

// Data Object in data set
struct DataObject
{
	EyePos eye;
	cv::Mat image;
	std::string filename;
};

constexpr EyePos InvalidEyePos{ -1,-1,-1,-1 };