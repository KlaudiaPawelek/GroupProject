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

Ptr<ml::ANN_MLP> ann;

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


Mat removeLeftBorder(Mat input)
{
	Mat tmp = Mat(input.size().height, input.size().width - 2, input.type());
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 2; j < input.cols; j++)
		{
			tmp.at<uchar>(i, j - 2) = input.at<uchar>(i, j);
		}
	}
	return tmp;
}
Mat ExtendRect(Rect& r, Mat sourceImage)
{
	Point tl = r.tl(); //top left corner Point
	Point br = r.br(); //bottom right corner Point
	int xIncrease = cvRound(r.width*0.15);
	int yIncrease = cvRound(r.height*0.15);
	if (tl.x - xIncrease > 0)
		tl.x -= xIncrease;
	else
		tl.x = 0;
	if (tl.y - yIncrease > 0)
		tl.y -= yIncrease;
	else
		tl.y = 0;
	if (br.x + xIncrease < sourceImage.cols)
		br.x += xIncrease;
	else
		br.x = sourceImage.cols;
	if (br.y + yIncrease < sourceImage.rows)
		br.y += yIncrease;
	else
		br.y = sourceImage.rows;
	r = Rect(tl, br);
	Mat RectPart = sourceImage(Range(tl.y, br.y), Range(tl.x, br.x));
	return RectPart;

}

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

vector<pair<int, Point>> findEdgeBubbles(Mat img,vector<Point> haarDetectedCenters)
{
	bool nextCont = false;
	vector<pair<int, Point>> circles;
	float annResult=0;
	int noOfFeatures = 3603;
	Mat testSample = Mat(1, noOfFeatures, CV_32FC1, Scalar(0));
	//Mat img = imread(argv[argP], IMREAD_GRAYSCALE);
	Mat imgCleanCopy = img.clone();
	Mat lbrImg = img.clone();

	Mat o = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
	Mat poly = o.clone();
	Point2f circleCenter;
	float circleRadius;

	Mat contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
	Mat singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));

	Mat out;
	vector<vector<Point>> contours;
	vector<vector<Point>> circleContour;
	vector<Point> polygonContour;

	//some image preprocessing
	Mat r2p = r2pTransform(lbrImg);
	out = r2p.clone();
	threshold(out, out, 120, 255, THRESH_BINARY);
	Mat cannyOut;
	Canny(out, cannyOut, 50, 200, 5);
	findContours(cannyOut, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		int contourPointOnEdgeCounter = 0;
		if (contourArea(contours[i]) > 20)
		{
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].x == (lbrImg.cols - 1) || contours[i][j].x == 0 || contours[i][j].y == 0 || contours[i][j].y == (lbrImg.rows - 1))
				{
					contourPointOnEdgeCounter++;
					if (contourPointOnEdgeCounter >= 1)
					{
						approxPolyDP(contours[i], polygonContour, 1, true);
						minEnclosingCircle(polygonContour, circleCenter, circleRadius);
						circle(singleContourImage, circleCenter, circleRadius, Scalar(255));
						findContours(singleContourImage, circleContour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

						Rect r = minAreaRect(circleContour[0]).boundingRect();
						for (int k = 0; k < haarDetectedCenters.size(); k++) //filtering out circles which are already detected
						{
							if (haarDetectedCenters[k].inside(r))
							{
								nextCont = true;
								break;
							}
						}
						if (nextCont)
						{
							nextCont = false;
							break;
						}
						Mat RectPart = ExtendRect(r, lbrImg);
						resize(RectPart, RectPart, Size(60, 60));

						testSample.at<float>(0, 0) = (float)circleCenter.x;
						testSample.at<float>(0, 1) = (float)circleCenter.y;
						testSample.at<float>(0, 2) = (float)circleRadius;
						int p = 3;
						for (int k = 0; k < RectPart.rows; k++)
						{
							for (int l = 0; l < RectPart.cols; l++)
							{
								testSample.at<float>(0, p) = (float)RectPart.at<uchar>(k, l) / 255;
								p++;
							}

						}
						annResult = ann->predict(testSample);

						if (annResult == 0)
						{
							if (circleCenter.x >= 0 && circleCenter.y >= 0 && circleCenter.x <= lbrImg.rows && circleCenter.y <= lbrImg.cols)
							{
								circles.push_back(make_pair(circleRadius, circleCenter));
							}
						}
						break;
					}
				}
			}

		}
		singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
	}
	return circles;
}

//Lens Cleanign implemented based on Karols code
vector<Mat> LensCleaning(int argc, const char **argv)
{
	vector<Mat> returnVMat; //creating vector of Mat which will store images after lens cleaning
	if (argc > 30) //lens cleaning is performed only if there is sufficient number of images provided
	{
		auto begin = chrono::system_clock::now();
		Mat image = imread(argv[5], IMREAD_GRAYSCALE); //reading first image
		//Mat r2pImage = r2pTransform(image); //performing r2pow and log transform
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
		imageMean = sumImage / (argc-5); //divding sum image by number of all images-> calculating mean of each pixel
		imageMean.convertTo(imageMean, CV_8UC1); //convertion back to 8bit image

		int avgGrayLevel = mean(imageMean)[0]; //average gray level value required to maitain the same gray level after noise removal
		//avgGrayLevel=sum(imageMean)[0]; //summing all pixels to one value
		//avgGrayLevel = avgGrayLevel / (image.rows*image.cols); //dividing sumed pixels values by number of pixels
		
		auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - begin;
		for (int i = 5; i < argc; i++)
		{
			Mat out = imread(argv[i], IMREAD_GRAYSCALE); //reading first image
			for (int j = 0; j < out.rows; j++)
			{
				for (int l = 0; l < out.cols; l++)
				{
					out.at<uchar>(j, l) = out.at<uchar>(j, l)- imageMean.at<uchar>(j, l)+avgGrayLevel;
				}
			}
			returnVMat.push_back(out); //assigning cleaned image to return vector
		}
	}
	return returnVMat; //returning images in vector
}

int main(int argc, const char **argv) {

	// Instructions for user
	//cout << "n - Next image (automatically saving image and all your changes" << endl;
	vector<int> saveParams;
	saveParams.push_back(IMWRITE_PNG_COMPRESSION);
	saveParams.push_back(9);

	ann = ml::ANN_MLP::load(argv[4]);

	if (!ann->isTrained())
	{
		cout << "Network not trained!" << endl;
		return -1;
	}

	calibration_factor = atof(argv[1]);  // calibration factor defined by user
	name_of_classifier = argv[3];        // name of classifier that should be used (for example Haar5.xml)
	number_of_drops = 0;                 // counter of droplets on all images
	vector<Mat> imagesAfterLC;
	auto start = chrono::system_clock::now();
	imagesAfterLC= LensCleaning(argc, argv);

										 // Loop through all images in the input folder, it starts from 5 because path to images is 5th argument on the command window
	for (int k = 5; k < argc; k++) 
	{

		path = argv[2];                                        // path to output folder
		name_of_file = path + "\\outputFile.txt";       // creating file with diameters in pixel unit
		keepProcessing = true;

		Mat imgLC;

		if(imagesAfterLC.size())
		imgLC = removeLeftBorder(imagesAfterLC[k - 5]);

		img = removeLeftBorder(imread(argv[k], 0));    // loading the image in grayscale (classifier was trained on greyscale images)
		img1 = imread(argv[k], 1);   // loading the same image in RGB (for user operations)

		Mat imgLCr2p;
		Mat imgr2p = r2pTransform(img);

		if(imagesAfterLC.size())
			imgLCr2p = removeLeftBorder(r2pTransform(imgLC));


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

			if (imagesAfterLC.size())
			cascade.detectMultiScale(imgLCr2p, dropsLC, 1.05, 3, 0, Size(5, 5), Size(570, 570));

			temp = k - 4;// number of image

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

				Mat circlePart = ExtendRect(r, imgr2p);

				vector<Vec3f> circlesDetected;
				HoughCircles(circlePart, circlesDetected, HOUGH_GRADIENT, 1, 1, 180, 80, radius*0.8, radius*1.2);

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
					for (int m = 0; m < dropsLC.size(); m++) 
					{
						// calculating the center of droplet
						Point center(dropsLC[m].x + dropsLC[m].width*0.5, dropsLC[m].y + dropsLC[m].height*0.5 + 3);
						if (center.x > 0.9*(newCenter.x + (r.tl().x)) && center.x < 1.1*(newCenter.x + (r.tl().x)) && center.y>0.9*(newCenter.y + (r.tl().y)) && center.y < 1.1*(newCenter.y + (r.tl().y)))
						{
							radius= cvRound(abs(dropsLC[m].width*0.5));
							break;
						}
					}
					circle(img1, Point(newCenter.x + (r.tl().x), newCenter.y + (r.tl().y)), radius, Scalar(58, 71, 244), 2); //drawing circle using center point of hough and radius of Haar
					centers.push_back(Point(newCenter.x + (r.tl().x), newCenter.y + (r.tl().y)));
					radiuses.push_back(radius);

				}
				else
				{
					centers.push_back(center);
					radiuses.push_back(radius);
					circle(img1, center, radius, Scalar(58, 71, 244),2);
				}

			}
			Mat NN;
			if (!imgLC.empty())
			{
				NN = imgLC;
			}
			else
			{
				NN = img;
			}
			
			vector<pair<int, Point>> edge = findEdgeBubbles(NN,centers);
			for (int i = 0; i < edge.size(); i++)
			{
				circle(img1, edge[i].second, edge[i].first, Scalar(58, 71, 244),2);
			}

			while (keepProcessing) 
			{
				//imshow(argv[k], img1);
				//key = waitKey(20);
				key = 'n';

				// Going to the next image or finishing programme if current image is the last one
				if (key == 'n') {
					imwrite(tempPath, img1,saveParams);
					keepProcessing = false;
					//destroyAllWindows();
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