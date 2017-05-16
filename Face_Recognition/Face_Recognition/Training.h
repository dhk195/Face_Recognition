#pragma once
#ifndef __TRAINING__HEADER__
#define __TRAINING__HEADER__
/****************************************************
* Filename: Training.h
* Author: Joshua Yuri M. Requioma
* Purpose: To create a file that holds different 
*	functions to help recognize faces. 
* Ver: 0.1
* Date created: May 12, 2017
****************************************************/

//OpenCV Libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face/facerec.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv::face;
using namespace cv;
using namespace std;

//Image files are located in the csv file where the program is "trained" to collect and compare.
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid file was given, please check the given filename";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int recognize_face(Mat image) {
	//Get the images
	vector<Mat>	data;
	vector<int>	labels;
	String csv_path = "csv_data.csv";
	//Displays original picture
	imshow("Original", image);

	try
	{
		read_csv(csv_path, data, labels);
	}
	catch (cv::Exception& e)
	{
		cout << "Error opening file: " << csv_path << "Reason: " << e.msg << endl;
		exit(1);
	}
	int height = data[0].rows;

	image = data[data.size() - 1];
	int testLabel = labels[labels.size() - 1];
	data.pop_back();
	labels.pop_back();
	Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(data, labels);
	model->setThreshold(0.0);
	int predictedLabel = model->predict(image);
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
		model->getRadius(),
		model->getNeighbors(),
		model->getGridX(),
		model->getGridY(),
		model->getThreshold());
	cout << model_info << endl;
	vector<Mat> histograms = model->getHistograms();
	cout << "Size of the histograms: " << histograms[0].total() << endl;
	imshow("Detected", image);
	waitKey(0);
	return 0;
}

#endif // !__TRAINING__HEADER__
