#include <opencv2/highgui.hpp>
#include <iostream>
#include "TrainDataSet.h"


int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cout << "Usage: mytrain [SrcCount] [ModelPath] [DataPath]\n";
		std::cout << "SrcCount: The number of source image imported to train\n";
		std::cout << "ModelPath: The path of extracted model file\n";
		std::cout << "DataPath: The path of training data set\n";
		return -1;
	}
	int srcCount = std::stoi(argv[1]);
	std::string modelPath = argv[2];
	std::string dataPath = argv[3];

	TrainDataSet data_set;
	data_set.loadDataSet(dataPath, srcCount);
	data_set.train();
	data_set.saveModel(modelPath);

	auto eigenFaces = data_set.outputEigenFace(srcCount > 10 ? 10 : srcCount);
	cv::Mat superEigenFace = eigenFaces[0];
	for (int i = 1; i < 5 && i < eigenFaces.size(); ++i)
	{
		cv::hconcat(superEigenFace, eigenFaces[i], superEigenFace);
	}
	if (eigenFaces.size() >= 5) {
		cv::Mat superEigenFace2 = eigenFaces[5];
		for (int i = 6; i < eigenFaces.size(); ++i)
		{
			cv::hconcat(superEigenFace2, eigenFaces[i], superEigenFace2);
		}
		cv::Mat black(eigenFaces[0].rows, eigenFaces[0].cols, CV_8UC1, cv::Scalar{ 0,0,0 });
		for (int i = eigenFaces.size(); i < 10; ++i)
		{
			cv::hconcat(superEigenFace2, black, superEigenFace2);
		}
		cv::vconcat(superEigenFace, superEigenFace2, superEigenFace);
	}

	cv::imshow("Eigenfaces", superEigenFace);


	//data_set.saveAllImages(".\\images_face\\");
	//data_set.saveAllRawImages(".\\images_raw\\");

	cv::waitKey();

}
