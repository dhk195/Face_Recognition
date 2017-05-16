// Face_Recognition.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "Training.h"
using namespace cv;
using namespace std;
int main()
{
	Mat image = imread("testcases/Steve.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Call function here
	recognize_face(image);	
	waitKey(0);
	return 0;
}
