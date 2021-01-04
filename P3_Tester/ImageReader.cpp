#include "ImageReader.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <io.h>



cv::Mat loadImage(const std::string& filename) noexcept
{
	cv::Mat imgmat = cv::imread(filename + ".pgm", cv::IMREAD_GRAYSCALE);

	return imgmat;
}

EyePos loadEyePos(const std::string& filename)
{
	std::fstream eps(filename + ".eye", std::ios::in);
	if (!eps.is_open())
		return InvalidEyePos;
	EyePos ep;
	std::string temp;

	// .eye format:
	// #LX 	LY	RX	RY
	// 232 	110	161	110
	std::getline(eps, temp);
	eps >> ep.RightX >> ep.RightY >> ep.LeftX >> ep.LeftY;

	eps.close();

	return ep;
}

DataObject ImageReader::loadFile(const std::string& filename)
{
	DataObject obj;

	obj.image = loadImage(filename);
	obj.eye = loadEyePos(filename);
	obj.filename = filename;

	return obj;
}

std::vector<DataObject> ImageReader::loadDataSet(const std::string& dir, int max)
{
	std::vector<DataObject> dataSet;

	// windows io function
	_finddata_t findData;
	auto hFile = _findfirst((dir + "*.pgm").c_str(), &findData);

	int count = 0;

	do
	{
		// confine the maximum size of the data set
		if (count == max) break;
		count++;

		std::string imagePath = dir + findData.name;
		// discard ".pgm"
		imagePath = imagePath.substr(0, imagePath.size() - 4);
		DataObject obj = loadFile(imagePath);

		// cv::imread doesn't throws exception, but the return mat is empty
		if (obj.image.empty())
		{
			std::cout << "Cannot open data " + imagePath + ".pgm\n";
			continue;
		}
		// when no eye file found, obj.eye will be set to InvalidEyePos
		if (obj.eye == InvalidEyePos)
		{
			std::cout << "Cannot open data " + imagePath + ".eye\n";
			continue;
		}
		dataSet.push_back(obj);

	} while (_findnext(hFile, &findData) == 0);

	_findclose(hFile);

	if (dataSet.empty())
	{
		std::cout << "Cannot find any data in " + dir << std::endl;
	}

	return dataSet;
}

void ImageReader::writeDataSet(const std::string& dir, const std::vector<DataObject>& dataset)
{
	for (int i = 0; i < dataset.size(); ++i)
	{
		cv::imwrite(dir + std::to_string(i) + ".jpg", dataset[i].image);
	}
}
