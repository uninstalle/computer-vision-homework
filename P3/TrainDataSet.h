#pragma once
#include <opencv2/imgproc.hpp>
#include <vector>
#include "DataStruct.h"


class TrainDataSet
{
	std::vector<DataObject> raw;
	std::vector<DataObject> data;
	cv::Mat avgMat;
	// pixel_num * image_num
	cv::Mat diffMat;
	// image_num * image_num, this is not the true covMat, but a reversely multiplied one
	cv::Mat covMat;
	cv::Mat eigenVector;
	cv::Mat tEigenVector;
	std::vector<cv::Mat> trainingValue;

	/*
	 * Preprocess the raw set, recognize faces and generate a new data set with the same size sub image.
	 */
	void preprocess(bool useCasClassifier = true);

	/*
	 * Get the average image, every pixel of which is the average value of the corresponding pixels of all data set images.
	 */
	cv::Mat findAverageImage() const;

	/*
	 * Calculate the difference vector of all images in the data set. Every pixel of the image will subtract the corresponding pixel of average image.
	 */
	cv::Mat calDiffMat() const;

	/*
	 * Calculate the covariant mat from member diffMat.
	 */
	cv::Mat calCovMat() const;

	/*
	 * Calculate the eigen vector from member covMat;
	 */
	cv::Mat calEigenVector() const;

	void getTrainingValue();
public:
	/*
	 * Load all images files in dir, save to raw set.
	 * All source images should be gray scaled.
	 * @param dir data set directory, end with \\
	 * @param max the maximum size of the data set, or infinite when max is negative
	 */
	void loadDataSet(const std::string& dir, int max = -1);

	void train();

	void recognizeImage(const DataObject& obj, bool useCasClassifier = true);

	void saveModel(const std::string& path);
	void loadModel(const std::string& path);

	/*
	 * Convert the eigen vector to image format and output it.
	 * @param num the number of output images, should be bigger than input image count
	 */
	std::vector<cv::Mat> outputEigenFace(int num);

	cv::Mat getAvgMat() { return avgMat; }
	cv::Mat getDiffMat() { return diffMat; }
	cv::Mat getCovMat() { return covMat; }
	cv::Mat getEigenVector() { return eigenVector; }

	// just don't use it
	void saveAllImages(const std::string& dir) const;
	// just don't use it
	void saveAllRawImages(const std::string& dir) const;
};

