#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

#include <stdlib.h>


using namespace cv;
using namespace std;

vector <Point> points;                         // vector of points; every three points create circle (LEFT MOUSE BUTTON action)
vector <Point> points_to_remove;               // vector of points;  these points are located inside droplets which should be removed (RIGHT MOUSE BUTTON action)
vector <Point> centers;                        // centers of detected bubbles
vector <int> radiuses;                        // radises of detected bubbles
vector<Rect> drops;                            // droplets detected by classifier
vector <double> diameters_in_mikrometers;      // diameters in mikrometers unit
double calibration_factor;
string name_of_classifier;
Point c(0, 0), S(0, 0), test(0, 0);
int radiusS = 0;
int difx, dify;

Mat img;
Mat img1;
CascadeClassifier cascade;
const string windowName = "Supervising";
int key;
int number_of_drops;

// File with Diameters in pixel unit
std::ofstream file;
string name_of_file;

//file with Diameters in mikrometers
std::ofstream file1;
string name_of_file1;


bool keepProcessing;
string path;                     // path to output folder
int temp;

double d2, d3, SMD;              // to calculate Sauter mean diameter


								 // function calculates center and radius of circle having three points on the circumference
void calculateCenter(Point P1, Point P2, Point P3)
{
	// coordinates of the first point
	int p1x = P1.x;
	int p1y = -P1.y;

	// coordinates of the second point
	int p2x = P2.x;
	int p2y = -P2.y;

	// coordinates of the third point
	int p3x = P3.x;
	int p3y = -P3.y;

	int p1x2 = P1.x*P1.x;
	int p1y2 = P1.y*P1.y;

	int p2x2 = P2.x*P2.x;
	int p2y2 = P2.y*P2.y;

	int p3x2 = P3.x*P3.x;
	int p3y2 = P3.y*P3.y;

	// calculating coordinates of the circle's center
	S.y = 0.5 * ((-p1x * p3x2 - p1x * p3y2 + p1x * p2x2 + p1x * p2y2 + p2x * p3x2 + p2x * p3y2 - p2x2 * p3x - p2y2 * p3x + p1x2 * p3x - p1x2 * p2x + p1y2 * p3x - p1y2 * p2x) / (p1y * p3x - p1y * p2x - p2y * p3x - p3y * p1x + p3y * p2x + p2y * p1x));
	S.x = (p3x2 - p2x2 + p3y2 - p2y2 + 2 * S.y*p2y - 2 * S.y*p3y) / (2 * (p3x - p2x));

	// calculating radius of the circle
	radiusS = sqrt((p1x - S.x)*(p1x - S.x) + (p1y - S.y)*(p1y - S.y));
	S.y *= -1;
}


// function supports mouse events
void supervisedCircles(int event, int x, int y, int flags, Mat* img)
{
	int row = y; // y-axis is image rows (down the side)
	int col = x; // x-axis is image columns (along the top)

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN: /// left mouse click

		c.x = x;
		c.y = y;
		points.push_back(c);                               // adding points to the vector 
		circle(*img, Point(x, y), 5, Scalar(0, 255, 0), 2); // drawing little, green circles on the image
		break;


	case CV_EVENT_RBUTTONDOWN: // right mouse click

		test.x = x;
		test.y = y;
		points_to_remove.push_back(test);                 // adding points to the vector
		circle(*img, Point(x, y), 10, Scalar(0, 0, 255)); // drawing red circles on the image
		break;
	}
}
/*******************************************************************/

void LensCleaning(int argc, const char **argv)
{
	if (argc > 30) //lens cleaning is performed only if there is sufficient number of images provided
	{
		Mat image = imread(argv[4], IMREAD_GRAYSCALE);

		image.convertTo(image, CV_16UC1);
		Mat sumImage = Mat(image.size(), CV_16UC1,Scalar(0));
		for (int imageIterator = 5; imageIterator < argc; imageIterator++)
		{
			add(sumImage,image,sumImage);
			image = imread(argv[imageIterator], IMREAD_GRAYSCALE);
			image.convertTo(image, CV_16UC1);
		}
		Mat imageMean = Mat(image.size(), CV_16UC1,Scalar(0));
		imageMean = sumImage / 100;
		
		int avgGrayLevel = 0;
		avgGrayLevel=sum(imageMean)[0];
		cout << avgGrayLevel << endl;
		avgGrayLevel = avgGrayLevel / (image.rows*image.cols);
		cout << "image loaded, parameters for lens cleaning calculated";
		imageMean.convertTo(imageMean, CV_8UC1);
		for (int i = 4; i < argc; i++)
		{
			Mat out = imread(argv[i], IMREAD_GRAYSCALE);
			//out.convertTo(out, CV_16UC1);
			imageMean.convertTo(imageMean, CV_8UC1); 
			//subtract(out, imageMean, out);
			absdiff(out, imageMean, out);
			out = out + avgGrayLevel;
			out.convertTo(out, CV_8UC1);
			imshow("LensCleaning", out);
			waitKey();				//WORKS PRETTY WELL BY FAR. NEED TO CHECK IF CONVERSION OF imageMEan to uchar type will be good too.
		}
	}
}

int main(int argc, const char **argv) {

	// Instroctions for user
	//cout << "Choose option that you need: " << endl;
	//cout << "To add your droplet to detected droplets choose three points on the perimeter using left mouse button and submit using 'd' " << endl;
	//cout << "To remove droplet use right mouse button: click inside all droplets that should be removed and submit your choice by clicking 'e'" << endl;
	//cout << "n - Next image (automatically saving image and all your changes" << endl;
	//cout << "r - refresh (you can check what will be saved)" << endl;


	calibration_factor = atof(argv[1]);  // calibration factor defined by user
	name_of_classifier = argv[3];        // name of classifier that should be used (for example Haar5.xml)
	number_of_drops = 0;                 // counter of droplets on all images

	LensCleaning(argc, argv);

										 // Loop through all images in the input folder, it starts from 4 because path to images is 5th argument on the command window
	for (int k = 4; k < argc; k++) {

		//cout << "Argc" << argc << endl;

		path = argv[2];                                        // path to output folder
		name_of_file = path + "\\diametersInPixels.txt";       // creating file with diameters in pixel unit
		name_of_file1 = path + "\\diametersInMikroMeters.txt"; // creating file with diameters in mikrometers
		keepProcessing = true;

		img = imread(argv[k], 0);    // loading the image in grayscale (classifier was trained on greyscale images)
		img1 = imread(argv[k], 1);   // loading the same image in RGB (for user operations)
	
		if (img.empty())
		{
			cout << "Failed to open image " << argv[k] << endl;
			return -1;
		}
		if (img1.empty())
		{
			cout << "Failed to open image " << argv[k] << endl;
			return -1;
		}
	 // If this is the first turn of the loop, programme save information about calibration factor
		if (k == 4) {
			file.open(name_of_file, std::ofstream::app);
			file << "Calibration factor: " << calibration_factor << " mikro meters/pixel" << endl;
			file.close();

			file1.open(name_of_file1, std::ofstream::app);
			file1 << "Calibration factor: " << calibration_factor << " mikro meters/pixel" << endl;
			file1.close();
		}


		if (cascade.load(name_of_classifier/*name of classifier*/) /*loading the cascade classifier*/) {

			// Function to detect droplets using Cascade Classifier; detected droplets are returned as a list of rectangles
			// vector "drops" stores information about rectangles coordinates
			cascade.detectMultiScale(img, drops, 1.05, 3, 0, Size(5, 5), Size(570, 570));

			temp = k - 3;// number of image

						 // converting variable "temp" into string type
			std::ostringstream sss;
			sss << temp;
			string str1 = sss.str();

			// Creating paths and names of images
			string tempPath = path;
			string name_of_image = "img" + str1;
			name_of_image = name_of_image + ".jpg";
			tempPath = tempPath + "\\";
			tempPath = tempPath + name_of_image;

			file.open(name_of_file, std::ofstream::app);
			if (file.is_open())
			{
				file << "Name of image: " << name_of_image << endl;
				file.close();
			}
			else
			{
				cout << "Failed to open file " << name_of_file << endl;
				return -1;
			}
			
			file1.open(name_of_file1, std::ofstream::app);
			if (file1.is_open())
			{
				file1 << "Name of image: " << name_of_image << endl;
				file1.close();
			}
			else
			{
				cout << "Failed to open file " << name_of_file1 << endl;
				return -1;
			}

			// iteration through all the vector with detected droplets
			for (int i = 0; i < drops.size(); i++) {
				Rect r = drops[i];
				// calculating the center of droplet
				Point center(drops[i].x + drops[i].width*0.5, drops[i].y + drops[i].height*0.5 + 3);
				// calculating radius of droplets
				int radius = abs(drops[i].width*0.5);
				int diameter = 2 * radius;

				// Adding calculated informations into vectors
				centers.push_back(center);
				radiuses.push_back(radius);

				// Drawing detected droplets
				circle(img1, center, radius, Scalar(255, 0, 0), 2);

			}

			//namedWindow(windowName, WINDOW_AUTOSIZE);
			//setMouseCallback("Supervising", (CvMouseCallback)supervisedCircles, &img1);


			while (keepProcessing) {
				//imshow(windowName, img1);
				//key = waitKey(20);

				key = 'n';
				// Drawing droplets marked by user
				if (key == 'd') {
					int numberOfCircles = floor(points.size() / 3);

					for (int i = 0; i < numberOfCircles; i++) {
						calculateCenter(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
						circle(img1, S, radiusS, Scalar(255, 0, 0), 2);
						centers.push_back(S);
						radiuses.push_back(radiusS);
					}
					points.clear();
					img1 = imread(argv[k], 1);
					for (int i = 0; i < centers.size(); i++) {
						circle(img1, centers[i], radiuses[i], Scalar(255, 0, 0), 2);

					}

				}

				// refreshing the image
				if (key == 'r') {
					img1 = imread(argv[k], 1);
					for (int i = 0; i < centers.size(); i++) {
						circle(img1, centers[i], radiuses[i], Scalar(255, 0, 0), 2);

					}
				}

				// removing droplets marked by user
				if (key == 'e') {
					bool keepProcessing = 1;
					int n = 0;
					int distance = 0;
					int check = 0;
					for (int i = 0; i < points_to_remove.size(); i++)
					{
						keepProcessing = 1;
						n = 0;
						while (keepProcessing) {
							distance = (points_to_remove[i].x - centers[n].x)*(points_to_remove[i].x - centers[n].x) + (points_to_remove[i].y - centers[n].y)*(points_to_remove[i].y - centers[n].y);
							check = radiuses[n] * radiuses[n] - distance;
							if (check > 0)
							{
								centers.erase(centers.begin() + n);
								radiuses.erase(radiuses.begin() + n);
								keepProcessing = 0;
								n = 0;
							}
							else
								n = n + 1;

						}

					}


					points_to_remove.clear();
					img1 = imread(argv[k], 1);
					for (int i = 0; i < centers.size(); i++) {
						circle(img1, centers[i], radiuses[i], Scalar(255, 0, 0), 2);

					}

				}


				// Going to the next image or finishing programme if current image is the last one
				if (key == 'n') {
					imwrite(tempPath, img1);
					keepProcessing = false;
				}

				//imshow(windowName, img1);
				//string fName=to_string(k);
				//
				//imwrite(fName+".jpg", img1);
			}


			// Saving information from current image in files
			file.open(name_of_file, std::ofstream::app);
			file << "Droplets sizes in pixels: " << endl;
			for (int i = 0; i < radiuses.size(); i++) {
				file << 2 * radiuses[i];
				file << std::endl;
			}
			file.close();

			file1.open(name_of_file1, std::ofstream::app);
			file1 << "Droplets sizes in mikro meters : " << endl;
			for (int i = 0; i < radiuses.size(); i++) {
				file1 << calibration_factor * 2 * radiuses[i];
				file1 << std::endl;
				diameters_in_mikrometers.push_back(calibration_factor * 2 * radiuses[i]);
			}
			file1.close();
			number_of_drops = number_of_drops + radiuses.size();

		}
		// Cleaning all the vectors
		centers.clear();
		radiuses.clear();
		points_to_remove.clear();
		points.clear();
		drops.clear();
	}

	// Calculating  Sauter mean diameter for all droplets on all images
	d3 = 0;
	d2 = 0;
	SMD = 0;
	for (int i = 0; i < diameters_in_mikrometers.size(); i++) {
		d3 = d3 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
		d2 = d2 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
	}
	SMD = d3 / d2;

	// Sauter Mean Diameter in file : "diametersInMikroMeters.txt"
	file1.open(name_of_file1, std::ofstream::app);
	file1 << "Number of all droplets:  " << number_of_drops << endl;
	file1 << "Sauter mean diameter d32 =  " << SMD << endl;
	file1.close();

	file.open(name_of_file, std::ofstream::app);
	file << "Number of all droplets:  " << number_of_drops << endl;
	file.close();
	cout << "Program finished without errors" << endl;
	return 0;
}