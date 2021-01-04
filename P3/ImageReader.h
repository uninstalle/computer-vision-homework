#pragma once
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "DataStruct.h"


class ImageReader
{
public:
	// file name only, suffix is pgm and eye by default.
	static DataObject loadFile(const std::string& filename);
	// load and create DataObject set with given max size, infinity when negative
	static std::vector<DataObject> loadDataSet(const std::string& dir, int max = -1);
	// why does a reader have a write function?
	static void writeDataSet(const std::string& dir, const std::vector<DataObject>& dataset);
};
