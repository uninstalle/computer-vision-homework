
#include <iostream>
#include <string>

#include "VideoMaker.h"

using std::string;
using std::cout;

int main(int argc, char** argv)
{
	string dir;
	if (argc == 1)
		dir = ".";
	else
		dir = argv[1];
	if (dir.back() == '\\') dir.erase(dir.length() - 1);

	VideoMaker* ass;
	if (argc == 4)
		ass = new VideoMaker(dir, std::stoi(argv[2]), std::stoi(argv[3]));
	else
		ass = new VideoMaker(dir);

	ass->writeNewVideo();


	return 0;
}