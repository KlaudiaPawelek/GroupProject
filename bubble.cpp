#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

#include <stdlib.h>
#include <chrono>

using namespace cv;
using namespace std;

vector <Point> points;                         // vector of points; every three points create circle (LEFT MOUSE BUTTON action)
vector <Point> points_to_remove;               // vector of points;  these points are located inside droplets which should be removed (RIGHT MOUSE BUTTON action)
vector <Point> centers;                        // centers of detected bubbles
vector <int> radiuses;                        // radises of detected bubbles
vector<Rect> drops;                            // droplets detected by classifier
vector<Rect> dropsLC;
vector <double> diameters_in_mikrometers;      // diameters in mikrometers unit
double calibration_factor;
string name_of_classifier;
int radiusS = 0;

Mat img;
Mat img1;
CascadeClassifier cascade;
int key;
int number_of_drops;

// File with output information
std::ofstream file;
string name_of_file;;


bool keepProcessing;
string path;                     // path to output folder
int temp;

double d2, d3, SMD;              // to calculate Sauter mean diameter

/*******************************************************************/
//Raise to power transform and logarithm transform.
Mat r2pTransform(Mat input)
{
	Mat r2powImage, logImage;

	//raise to power transform
	input.convertTo(r2powImage, CV_32F); //changing pixel value type (from uchar to float)
	r2powImage = r2powImage + 1; // dont know what is this :v
	pow(r2powImage, 5, r2powImage);
	normalize(r2powImage, r2powImage, 0, 255, NORM_MINMAX); //normalizing picture
	convertScaleAbs(r2powImage, r2powImage); //converting back to uchar from float

	//log transform
	r2powImage.convertTo(logImage, CV_32F); //conversion of r2powImage from uchar to float
	logImage = logImage + 1; //dont know what is this
	log(logImage, logImage); //performing log transform
	normalize(logImage, logImage, 0, 255, NORM_MINMAX); //normalizing picture
	convertScaleAbs(logImage, logImage); //converting back to uchar
	return logImage; //returning transformed image
}

//Lens Cleanign implemented based on Karols code
vector<Mat> LensCleaning(int argc, const char **argv)
{
	vector<Mat> returnVMat(argc-4); //creating vector of Mat which will store images after lens cleaning
	if (argc > 30) //lens cleaning is performed only if there is sufficient number of images provided
	{
		auto begin = chrono::system_clock::now();
		Mat image = imread(argv[4], IMREAD_GRAYSCALE); //reading first image
		Mat r2pImage = r2pTransform(image); //performing r2pow and log transform
		image = r2pImage.clone();
		image.convertTo(image, CV_16UC1); //conversion to 16bit image
		Mat sumImage = Mat(image.size(), CV_16UC1,Scalar(0)); //alocating Mat object which will store sum of pixels
		//looping through rest of images to add all images to one
		for (int imageIterator = 5; imageIterator < argc; imageIterator++)
		{
			add(sumImage,image,sumImage); //adding images
			image = imread(argv[imageIterator], IMREAD_GRAYSCALE); //reading next image
			image.convertTo(image, CV_16UC1); //convertion to 16bit image
		}

		Mat imageMean = Mat(image.size(), CV_16UC1,Scalar(0)); //alocating Mat object which will store average pixel value
		imageMean = sumImage / (argc-4); //divding sum image by number of all images-> calculating mean of each pixel
		imageMean.convertTo(imageMean, CV_8UC1); //convertion back to 8bit image

		int avgGrayLevel = 0; //average gray level value required to maitain the same gray level after noise removal
		avgGrayLevel=sum(imageMean)[0]; //summing all pixels to one value
		//cout << avgGrayLevel << endl;
		avgGrayLevel = avgGrayLevel / (image.rows*image.cols); //dividing sumed pixels values by number of pixels
		//cout << "image loaded, parameters for lens cleaning calculated";
		
		auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - begin;
		//cout << "Time :" << elapsed_seconds.count() << endl; //about 2 seconds
		for (int i = 4; i < argc; i++)
		{
			begin = chrono::system_clock::now();
			Mat out = imread(argv[i], IMREAD_GRAYSCALE); //reading first image
			//out.convertTo(out, CV_16UC1);
			imageMean.convertTo(imageMean, CV_8UC1); 
			//subtract(out, imageMean, out);
			absdiff(out, imageMean, out); //performing lens cleaning based on equation from Karols paper/code
			out = out + avgGrayLevel;
			out.convertTo(out, CV_8UC1);
			end = chrono::system_clock::now();
			elapsed_seconds = end - begin;
			bitwise_not(out, out); //
			returnVMat[i - 4] = out; //assigning cleaned image to return vector
			//cout << "Time :" << elapsed_seconds.count() << endl; //about 0.03 second
			//imshow("LensCleaning", out);
			//waitKey();
		}
	}
	return returnVMat; //returning images in vector
}

int main(int argc, const char **argv) {

	// Instroctions for user
	//cout << "Choose option that you need: " << endl;
	//cout << "To add your droplet to detected droplets choose three points on the perimeter using left mouse button and submit using 'd' " << endl;
	//cout << "To remove droplet use right mouse button: click inside all droplets that should be removed and submit your choice by clicking 'e'" << endl;
	//cout << "n - Next image (automatically saving image and all your changes" << endl;
	//cout << "r - refresh (you can check what will be saved)" << endl;
	vector<int> saveParams;
	saveParams.push_back(IMWRITE_PNG_COMPRESSION);
	saveParams.push_back(9);
	calibration_factor = atof(argv[1]);  // calibration factor defined by user
	name_of_classifier = argv[3];        // name of classifier that should be used (for example Haar5.xml)
	number_of_drops = 0;                 // counter of droplets on all images
	vector<Mat> imagesAfterLC(argc - 4);
	auto start = chrono::system_clock::now();
	imagesAfterLC= LensCleaning(argc, argv);

										 // Loop through all images in the input folder, it starts from 4 because path to images is 5th argument on the command window
	for (int k = 4; k < argc; k++) 
	{

		path = argv[2];                                        // path to output folder
		name_of_file = path + "\\outputFile.txt";       // creating file with diameters in pixel unit
		keepProcessing = true;

		Mat imgLC = imagesAfterLC[k - 4];
		Mat img1LC = imagesAfterLC[k - 4];

		img = imread(argv[k], 0);    // loading the image in grayscale (classifier was trained on greyscale images)
		img1 = imread(argv[k], 1);   // loading the same image in RGB (for user operations)
		Mat imgLCr2p;
		Mat imgr2p = r2pTransform(img);
		if(!imgLC.empty())
		imgLCr2p = r2pTransform(imgLC);

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


		if (cascade.load(name_of_classifier/*name of classifier*/) /*loading the cascade classifier*/) {

			// Function to detect droplets using Cascade Classifier; detected droplets are returned as a list of rectangles
			// vector "drops" stores information about rectangles coordinates
			cascade.detectMultiScale(img, drops, 1.05, 3, 0, Size(5, 5), Size(570, 570));

			if(!imgLC.empty())
			cascade.detectMultiScale(imgLC, dropsLC, 1.05, 3, 0, Size(5, 5), Size(570, 570));

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

			// iteration through all the vector with detected droplets
			for (int i = 0; i < drops.size(); i++) 
			{
				Rect r = drops[i];
				// calculating the center of droplet
				Point center(drops[i].x + drops[i].width*0.5, drops[i].y + drops[i].height*0.5 + 3);
				// calculating radius of droplets
				int radius = cvRound(abs(drops[i].width*0.5));
				int diameter = 2 * radius;
				Point tl = r.tl(); //top left corner Point
				Point br = r.br(); //bottom right corner Point
				int xIncrease = cvRound(drops[i].width*0.3);
				int yIncrease = cvRound(drops[i].height*0.3);
				if (tl.x - xIncrease > 0)
					tl.x -= xIncrease;
				else 
					tl.x = 0;
				if (tl.y - yIncrease > 0)
					tl.y -= yIncrease;
				else 
					tl.y = 0;
				if (br.x + xIncrease < img1.cols)
					br.x += xIncrease;
				else
					br.x = img1.cols;
				if (br.y + yIncrease < img1.rows)
					br.y += yIncrease;
				else 
					br.y = img1.rows;

				Rect circleRectangle;
				Mat circlePart = imgr2p(Range(tl.y,br.y), Range(tl.x, br.x));
				vector<Vec3f> circlesDetected;
				HoughCircles(circlePart, circlesDetected, HOUGH_GRADIENT, 1, 1, 180, 80, radius*0.7, radius*1.3);

				int cIt = 0;
				double newRadius = 0;
				double newRadiusAvg = 0;
				int maxRadiusIndx = 0;
				Point newCenter=Point(0,0);

				if (circlesDetected.size() && circlesDetected[cIt][1]!=0 && circlesDetected[cIt][1]!=0 && circlesDetected[cIt][2]!=0)
				{
					for (cIt; cIt < circlesDetected.size(); cIt++) //if Hough transform found new circles we sum center and radiuses to calculate average
					{
						newCenter.x += circlesDetected[cIt][0];
						newCenter.y += circlesDetected[cIt][1];
						newRadiusAvg += circlesDetected[cIt][2];

					}
					newCenter.x /= circlesDetected.size();
					newCenter.y /= circlesDetected.size();
					newRadiusAvg /= circlesDetected.size();
					for (int m = 0; m < dropsLC.size(); m++) {
						Rect r = dropsLC[m];
						// calculating the center of droplet
						Point center(dropsLC[m].x + dropsLC[m].width*0.5, dropsLC[m].y + dropsLC[m].height*0.5 + 3);
						if (center.x > 0.9*(newCenter.x + (tl.x)) && center.x < 1.1*(newCenter.x + (tl.x)) && center.y>0.9*(newCenter.y + (tl.y)) && center.y < 1.1*(newCenter.y + (tl.y)))
						{
							radius= cvRound(abs(dropsLC[m].width*0.5));
							break;
						}
					}
					circle(img1, Point(newCenter.x + (tl.x), newCenter.y + (tl.y)), radius, Scalar(0, 255, 0), 1); //drawing circle using center point of hough and radius of Haar
					centers.push_back(Point(newCenter.x + (tl.x), newCenter.y + (tl.y)));
					radiuses.push_back(radius);

				}
				else
				{
					centers.push_back(center);
					radiuses.push_back(radius);
					circle(img1, center, radius, Scalar(0, 255, 0), 1);
				}

			}

			while (keepProcessing) {
				//imshow(windowName, img1);
				//key = waitKey(20);
				key = 'n';

				// Going to the next image or finishing programme if current image is the last one
				if (key == 'n') {
					imwrite(tempPath, img1,saveParams);
					keepProcessing = false;
				}

			}


			// Saving information from current image in files. Mean diameter, min, max and SMD
			file.open(name_of_file, std::ofstream::app);
			file << "Nb : " << radiuses.size() << endl;
			double meanDiameter = 0;
			double maxDiameter = 0;
			double minDiameter=1000;
			vector<double> diamTmp;
			for (int m = 0; m < radiuses.size(); m++)
			{
				meanDiameter += radiuses[m] * 2;
				if (maxDiameter < radiuses[m] * 2)
				{
					maxDiameter = radiuses[m] * 2;
				}
				if (minDiameter > radiuses[m] * 2)
				{
					minDiameter = radiuses[m] * 2;
				}
				diamTmp.push_back(calibration_factor * 2 * radiuses[m]);
				diameters_in_mikrometers.push_back(calibration_factor * 2 * radiuses[m]);
			}
			double d3tmp = 0;
			double d2tmp = 0;
			double SMDtmp = 0;
			for (int d = 0; d < diamTmp.size(); d++) {
				d3tmp = d3tmp + diamTmp[d] * diamTmp[d] * diamTmp[d];
				d2tmp = d2tmp + diamTmp[d] * diamTmp[d];
			}
			SMDtmp = d3tmp / d2tmp;
			file << "SMD : " << SMDtmp << endl;
			file << "Mean : " << (calibration_factor * (meanDiameter/radiuses.size())) << endl;
			file << "Min : " << calibration_factor*minDiameter << endl;
			file << "Max : " << calibration_factor*maxDiameter << endl;
			file.close();

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
	double meanTotal=0;
	double minTotal=1000;
	double maxTotal=0;
	for (int i = 0; i < diameters_in_mikrometers.size(); i++) {
		d3 = d3 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
		d2 = d2 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
		meanTotal += diameters_in_mikrometers[i];
		if (maxTotal < diameters_in_mikrometers[i] )
		{
			maxTotal = diameters_in_mikrometers[i] ;
		}
		if (minTotal > diameters_in_mikrometers[i])
		{
			minTotal = diameters_in_mikrometers[i];
		}
	}
	SMD = d3 / d2;
	auto end = chrono::system_clock::now();

	// Sauter Mean Diameter in file : "diametersInMikroMeters.txt"
	file.open(name_of_file, std::ofstream::app);
	
	file << endl<< "NbAll : " << number_of_drops << endl;
	file << "SMDAll : " << SMD << endl;
	file << "MeanAll : " << (meanTotal / diameters_in_mikrometers.size()) << endl;
	file << "MinAll : " << minTotal << endl;
	file << "MaxAll : " << maxTotal << endl;
	file << "Time : " << chrono::duration_cast<chrono::seconds>(end - start).count()<<" sec"<<endl;
	file.close();

	cout << "Program finished without errors" << endl;
	return 0;
}