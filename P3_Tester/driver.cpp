#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include "TrainDataSet.h"

EyePos loadEP(const std::string& filename);

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: mytest [ModelPath] [optional:SrcImg] [optional:EyePosPath]\n";
		std::cout << "ModelPath: The path of extracted model file\n";
		std::cout << "SrcImg: The image needs to be recognized. You can also starts the program first\n";
		std::cout << "EyePosPath: The path of the eye position of input image. It is optional if you enable face cascade classifier, but cannot handle when classifier cannot find a face\n";
		return -1;
	}

	TrainDataSet data_set;
	cv::Mat img;
	EyePos ep = InvalidEyePos;

	std::string modelPath = argv[1];
	data_set.loadModel(modelPath);

	std::string imgPath;
	if (argc >= 3)
	{
		imgPath = argv[2];
		img = cv::imread(imgPath);
	}
	std::string eyePosPath;
	if (argc >= 4)
	{
		eyePosPath = argv[3];
		ep = loadEP(eyePosPath);
	}



	while (true) {

		DataObject obj;
		obj.image = img;
		obj.eye = ep;
		obj.filename = imgPath;
		if (!img.empty())
			data_set.recognizeImage(obj);

		std::cout << "Enter next image to recognize?\n";
		std::cout << "[SrcImg] [optional:EyePosPath]\n";
		std::cout << "SrcImg: The image needs to be recognized.\n";
		std::cout << "EyePosPath: The path of the eye position of input image. It is optional if you enable face cascade classifier, but cannot handle when classifier cannot find a face\n";
		std::cout << "\"quit\": To quit the program.\n";

		std::string cmd;
		std::getline(std::cin, cmd);
		if (cmd == "quit")
			break;
		imgPath = cmd.substr(0, cmd.find(' '));
		img = cv::imread(imgPath);
		if (cmd.find(' ') != std::string::npos)
		{
			cmd = cmd.substr(cmd.find(' ') + 1);
			ep = loadEP(cmd);
		}
		else ep = InvalidEyePos;

	}
}


EyePos loadEP(const std::string& filename)
{
	std::fstream eps(filename, std::ios::in);
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