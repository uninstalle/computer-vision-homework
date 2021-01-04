#include "TrainDataSet.h"
#include "ImageReader.h"
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


void TrainDataSet::loadDataSet(const std::string& dir, int max)
{
	raw = ImageReader::loadDataSet(dir, max);
}

void TrainDataSet::saveAllRawImages(const std::string& dir) const
{
	ImageReader::writeDataSet(dir, raw);
}

void TrainDataSet::saveAllImages(const std::string& dir) const
{
	ImageReader::writeDataSet(dir, data);
}


cv::Rect applyEyeFaceTemplate(const cv::Mat& image, const EyePos& eyePos)
{

	// eye-face template
	// L eye :59 75 R eye : 130 75 Face : 186 186

	cv::Point2d left(eyePos.LeftX, eyePos.LeftY), right(eyePos.RightX, eyePos.RightY);

	cv::Point2d mid((left.x + right.x) / 2, (left.y + right.y) / 2);
	// euclid dist
	double dist = std::sqrt(std::pow((right.x - left.x), 2) + std::pow((right.y - left.y), 2));
	int templateDist = 130 - 59;
	double rate = dist / templateDist;

	cv::Point2d leftTop(eyePos.LeftX - 59 * rate, eyePos.LeftY - 75 * rate),
		RightBottom(eyePos.RightX + (186 - 130) * rate, eyePos.RightY + (186 - 75) * rate);

	// fix out-of-range, while keeping the rect is a square

	if (leftTop.y < 0)
	{
		// no forehead in the image
		int oor = -leftTop.y;
		leftTop.y = 0;
		leftTop.x += oor / 2;
		RightBottom.x -= oor / 2;
	}

	if (RightBottom.y > image.rows)
	{
		int oor = RightBottom.y - image.rows;
		RightBottom.y = image.rows;
		leftTop.x += oor / 2;
		RightBottom.x -= oor / 2;
	}
	if (leftTop.x < 0)
	{
		int oor = -leftTop.x;
		leftTop.x = 0;
		leftTop.y += oor;
		RightBottom.x -= oor;
		RightBottom.y -= oor;
	}
	if (RightBottom.x > image.cols)
	{
		int oor = RightBottom.x - image.cols;
		RightBottom.x = image.cols;
		leftTop.x += oor;
		leftTop.y -= oor;
		RightBottom.y -= oor;
	}

	return cv::Rect(leftTop, RightBottom);
}

void TrainDataSet::preprocess(bool useCasClassifier)
{
	// face detection
	cv::CascadeClassifier faceCascade;
	faceCascade.load(".\\data\\haarcascade_frontalface_default.xml");
	if (faceCascade.empty())
	{
		std::cout << "Face cascade classifier not found. Will use eye-face template instead.\n";
		useCasClassifier = false;
	}


	int avgWidth = 0, avgHeight = 0;

	for (auto& obj : raw)
	{
		std::vector<cv::Rect> faces;

		if (useCasClassifier)
			faceCascade.detectMultiScale(obj.image, faces);

		// no face detected, unable to proceed
		if (faces.empty())
		{
			if (useCasClassifier)
				std::cout << "0 face detected at " + obj.filename << ", using eye-face template to generate face rect." << std::endl;

			cv::Rect face = applyEyeFaceTemplate(obj.image, obj.eye);
			faces.push_back(face);
		}


		// normally there should be only one face
		cv::Mat faceImage = obj.image(faces.at(0));

		avgWidth += faceImage.cols;
		avgHeight += faceImage.rows;

		// map EyePos to the sub mat coordination
		EyePos newEyePos = { obj.eye.LeftX - faces.at(0).x,
			obj.eye.LeftY - faces.at(0).y,
		obj.eye.RightX - faces.at(0).x ,
		obj.eye.RightY - faces.at(0).y };

		data.push_back(DataObject{ newEyePos,faceImage, obj.filename });
	}

	// it seems that the detected faces are always N * N, thus one variable should be enough
	avgWidth /= raw.size();
	avgHeight /= raw.size();
	cv::Size avgSize(avgWidth, avgHeight);

	// resize all images to the same size(averaged)
	for (auto& obj : data)
	{
		cv::Mat avgSizeImage(avgSize, CV_8UC1);
		cv::resize(obj.image, avgSizeImage, avgSize);
		double rate = static_cast<double>(avgSizeImage.rows) / obj.image.rows;
		obj.eye = { static_cast<int>(obj.eye.LeftX * rate),
			static_cast<int>(obj.eye.LeftY * rate),
			static_cast<int>(obj.eye.RightX * rate),
			static_cast<int>(obj.eye.RightY * rate) };
		obj.image = avgSizeImage;
	}

	// make the pixel value distribution more normal
	for (auto& obj : data)
	{
		cv::equalizeHist(obj.image, obj.image);
	}
}

cv::Mat TrainDataSet::findAverageImage() const
{
	cv::Mat avgMat(data.at(0).image.rows, data.at(0).image.cols, data.at(0).image.type());

	int rows = avgMat.rows;
	int cols = avgMat.cols;
	int matSize = rows * cols;

	// temp is a super long line generated from the mat
	// unsigned int should be enough? 32bit / 8bit = 2 ^ 24 possible image pixel sum upper bound
	auto temp = new unsigned[matSize] {};

	// calculate average pixel of the data set images
	for (auto& obj : data)
	{
		obj.image.forEach<uchar>([temp, cols](uchar pixel, const int* position)
			{
				// position[0] : row, position[1] : col
				temp[position[0] * cols + position[1]] += pixel;
			});
	}

	for (int i = 0; i < matSize; ++i)
		temp[i] /= data.size();

	// copy the average pixel value to avgMat
	avgMat.forEach<uchar>([temp, cols](uchar& pixel, const int* position)
		{
			pixel = temp[position[0] * cols + position[1]];
		});

	delete[] temp;
	return avgMat;
}

cv::Mat TrainDataSet::calDiffMat() const
{
	auto pAvgMat = avgMat.data;
	int rows = avgMat.rows;
	int cols = avgMat.cols;
	int matSize = rows * cols;

	// diffMat is [0,255] - [0,255] = [-255,255], CV_8SC1 will overflow/underflow, thus use CV_16SC1
	// but I guess using CV_8SC1 is fine, since the diffMat should be pretty small
	// ************************************************************
	// never mind, only float number supports vector multiply
	int diffCols = data.size();
	cv::Mat diffMat(matSize, diffCols, CV_32FC1);
	auto pDiffMat = diffMat.ptr<float>();

	for (int i = 0; i < data.size(); ++i)
	{
		data[i].image.forEach<uchar>([pDiffMat, pAvgMat, i, cols, diffCols](uchar pixel, const int* position)
			{
				int offset = position[0] * cols + position[1];
				pDiffMat[offset * diffCols + i] = static_cast<float>(pixel) - pAvgMat[offset];
			});
	}

	return diffMat;
}

cv::Mat TrainDataSet::calCovMat() const
{
	cv::Mat tDiff;
	cv::transpose(diffMat, tDiff);
	int m = data.size();
	// every pixel is the sum of m * diffMat.size * pixel * pixel.
	// reverse the order to decrease complexity
	cv::Mat covMat = tDiff * diffMat / m;

	return covMat;
}

cv::Mat TrainDataSet::calEigenVector() const
{
	cv::Mat eigenValue;
	cv::Mat eigenVector;
	cv::eigen(covMat, eigenValue, eigenVector);

	// turn the eigenVector of reversed covMat to that of the true covMat
	eigenVector = diffMat * eigenVector;

	return eigenVector;
}

std::vector<cv::Mat> TrainDataSet::outputEigenFace(int num)
{
	// eigenface images are less than input images
	assert(num <= data.size());

	// map the float value to unsigned char
	std::vector<cv::Mat> m(num);

	// global normalize can decrease the noise, but will also make the image pale.
	//cv::normalize(eigenVector, eigenVector, 0, 255, cv::NORM_MINMAX);

	for (int i = 0; i < num; ++i)
	{
		// the i-th vector
		cv::Mat vec = eigenVector.col(i);

		// local normalize
		cv::normalize(vec, vec, 0, 255, cv::NORM_MINMAX);

		m[i] = cv::Mat(avgMat.rows, avgMat.cols, CV_8UC1);
		auto p = m[i].data;
		vec.forEach<float>([p](float pixel, const int* position)
			{
				// row = pos, col = 0
				p[position[0]] = pixel;
			});
	}

	return m;
}

void TrainDataSet::getTrainingValue()
{
	// normalize the vector before calculating weight
	for (int i = 0; i < data.size(); ++i)
	{
		cv::Mat vec = tEigenVector.row(i);
		cv::normalize(vec, vec);
	}

	for (int i = 0; i < diffMat.cols; ++i)
	{
		auto diff = diffMat.col(i);
		cv::Mat weight = tEigenVector * diff;
		trainingValue.push_back(weight);
	}
}


void TrainDataSet::train()
{
	preprocess();
	avgMat = findAverageImage();
	diffMat = calDiffMat();
	covMat = calCovMat();
	eigenVector = calEigenVector();
	cv::transpose(eigenVector, tEigenVector);
	getTrainingValue();
}

void TrainDataSet::recognizeImage(const DataObject& obj, bool useCasClassifier)
{
	cv::Mat imageGray;
	cv::cvtColor(obj.image, imageGray, cv::COLOR_BGR2GRAY);

	// face detection
	cv::CascadeClassifier faceCascade;
	faceCascade.load(".\\data\\haarcascade_frontalface_default.xml");

	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(imageGray, faces);

	if (useCasClassifier)
		faceCascade.detectMultiScale(imageGray, faces);

	// no face detected, unable to proceed
	if (faces.empty())
	{
		if (useCasClassifier)
			std::cout << "0 face detected at " + obj.filename << ", using eye-face template to generate face rect." << std::endl;
		if (!(obj.eye == InvalidEyePos))
		{
			cv::Rect face = applyEyeFaceTemplate(imageGray, obj.eye);
			faces.push_back(face);
		}
		else
		{
			std::cout << "No valid eye pos, abort.\n";
			return;
		}
	}

	// normally there should be only one face
	cv::Mat faceImage = imageGray(faces.at(0));

	// it seems that the detected faces are always N * N, thus one variable should be enough

	cv::Size avgSize(avgMat.cols, avgMat.rows);

	// resize all images to the same size(averaged)

	cv::Mat avgSizeImage(avgSize, CV_8UC1);
	cv::resize(faceImage, avgSizeImage, avgSize);

	// make the pixel value distribution more normal
	cv::equalizeHist(imageGray, imageGray);

	// get diff mat
	auto pAvgMat = avgMat.data;
	int rows = avgMat.rows;
	int cols = avgMat.cols;
	int matSize = rows * cols;

	cv::Mat diff(matSize, 1, CV_32FC1);
	auto pDiffMat = diff.ptr<float>();

	avgSizeImage.forEach<uchar>([pDiffMat, pAvgMat, cols](uchar pixel, const int* position)
		{
			int offset = position[0] * cols + position[1];
			pDiffMat[offset] = static_cast<float>(pixel) - pAvgMat[offset];
		});
	// get weight
	cv::Mat weight = (tEigenVector * diff);

	// get dist
	double minDist = cv::norm(weight, trainingValue[0]);
	int closestImage = 0;
	for (int i = 1; i < trainingValue.size(); ++i)
	{
		double v = cv::norm(weight, trainingValue[i]);
		if (v < minDist)
		{
			minDist = v;
			closestImage = i;
		}
	}
	std::cout << "The most similar image is " << raw[closestImage].filename << ", with dist = " << minDist << std::endl;
	std::string shortName = raw[closestImage].filename.substr(raw[closestImage].filename.find_last_of('\\'));
	cv::putText(obj.image, "Similar:" + shortName, { 0,30 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 50,50,255 });
	cv::putText(obj.image, "Dist=" + std::to_string(minDist), { 0,60 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 50,50,255 });
	//
	cv::imshow("Source", obj.image);
	cv::imshow("Similar", raw[closestImage].image);
	cv::waitKey();
}

void TrainDataSet::loadModel(const std::string& path)
{
	cv::FileStorage fs;
	fs.open(path + "model.xml", cv::FileStorage::Mode::READ);
	if (!fs.isOpened())
	{
		std::cout << "Open model failed.\n";
		return;
	}
	int srcNum;
	fs["srcNum"] >> srcNum;
	fs["AvgMat"] >> avgMat;
	fs["TransEigenVector"] >> tEigenVector;
	for (int i = 0; i < srcNum; ++i)
	{
		cv::Mat tValue;
		fs["Training" + std::to_string(i)] >> tValue;
		trainingValue.push_back(tValue);
	}
	for (int i = 0; i < srcNum; ++i)
	{
		cv::Mat src;
		std::string name;
		fs["src" + std::to_string(i)] >> src;
		fs["srcName" + std::to_string(i)] >> name;
		DataObject obj;
		obj.image = src;
		obj.filename = name;
		raw.push_back(obj);
	}
	fs.release();
}

void TrainDataSet::saveModel(const std::string& path)
{
	cv::FileStorage fs;
	fs.open(path + "model.xml", cv::FileStorage::Mode::WRITE);
	if (!fs.isOpened())
	{
		std::cout << "Open model failed.\n";
		return;
	}
	fs << "srcNum" << static_cast<int>(raw.size());
	fs << "AvgMat" << avgMat;
	fs << "TransEigenVector" << tEigenVector;
	for (int i = 0; i < trainingValue.size(); ++i)
	{
		fs << "Training" + std::to_string(i) << trainingValue[i];
	}
	for (int i = 0; i < raw.size(); ++i)
	{
		fs << "src" + std::to_string(i) << raw[i].image;
		fs << "srcName" + std::to_string(i) << raw[i].filename;
	}
	fs.release();
}
