#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

using namespace cv;
using namespace std;

vector<pair<float,Point>> bubbles;
vector<pair<float, Point>> edgeBubbles;
vector <Point> centers;                        // centers of detected bubbles
vector <int> radiuses;                        // radises of detected bubbles
vector<Rect> drops;                            // droplets detected by classifier
vector<Rect> dropsLC;
vector <double> diameters_in_mikrometers;      // diameters in mikrometers unit
double calibration_factor;
string name_of_classifier;

vector<Mat> images;

vector<pair<int, int>> adaptiveThresholdParams = { make_pair(35,35),make_pair(5,15),make_pair(27,15),make_pair(51,71),make_pair(5,5) }; //size and C

Ptr<ml::ANN_MLP> ann;
int imgType=1;

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

//two first columns contain dark pixels so we remove those columns as it interrupts contours detection
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
Mat ExtendRect(Rect& r, Mat sourceImage,double extendFactor)
{
	Point tl = r.tl(); //top left corner Point
	Point br = r.br(); //bottom right corner Point
	int xIncrease = cvRound(r.width*extendFactor);
	int yIncrease = cvRound(r.height*extendFactor);
	if (tl.x - xIncrease >= 0)
		tl.x -= xIncrease;
	else
		tl.x = 0;
	if (tl.y - yIncrease >= 0)
		tl.y -= yIncrease;
	else
		tl.y = 0;
	if (br.x + xIncrease <= sourceImage.cols)
		br.x += xIncrease;
	else
		br.x = sourceImage.cols;
	if (br.y + yIncrease <= sourceImage.rows)
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

vector<pair<float, Point>> findEdgeBubbles(Mat img,vector<pair<float,Point>>& haarDetectedCenters)
{
	bool nextCont = false;
	vector<pair<float, Point>> circles;
	float annResult=0;
	int noOfFeatures = 3603;
	Mat testSample = Mat(1, noOfFeatures, CV_32FC1, Scalar(0));
	Mat lbrImg = removeLeftBorder(img);

	Point2f circleCenter;
	float circleRadius;

	Mat circleImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
	Mat contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
	Mat singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));

	vector<vector<Point>> contours;
	vector<vector<Point>> circleContour;
	vector<Point> polygonContour;

	//some image preprocessing
	Mat imgr2p = r2pTransform(lbrImg); //raise to power transform improved contrast
	Mat adptThresh;

	//performing adaptive thresholding using apropriate parameters. 
	adaptiveThreshold(imgr2p, adptThresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, adaptiveThresholdParams[imgType-1].first, adaptiveThresholdParams[imgType - 1].second);

	//finding contours in image which will be processed to detect circles
	findContours(adptThresh, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 50)
		{
			approxPolyDP(contours[i], polygonContour, 1, true);
			minEnclosingCircle(polygonContour, circleCenter, circleRadius);
			if (circleCenter.x - 2 * circleRadius <= 0 || circleCenter.x + 2 * circleRadius >= imgr2p.cols || circleCenter.y - 2 * circleRadius <= 0 || circleCenter.y + 2 * circleRadius >= imgr2p.rows)
			{
				if (circleCenter.x >= 0 && circleCenter.y >= 0)
				{
					circle(singleContourImage, circleCenter, circleRadius, Scalar(255));
					findContours(singleContourImage, circleContour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
					Rect r = minAreaRect(circleContour[0]).boundingRect();

					if (!r.empty())
					{
						Mat circleCandidate;
						circleCandidate = ExtendRect(r, imgr2p, 0.1);

						if (mean(circleCandidate)[0] > 50)
						{
							resize(circleCandidate, circleCandidate, Size(60, 60));
							testSample.at<float>(0, 0) = (float)circleCenter.x;
							testSample.at<float>(0, 1) = (float)circleCenter.y;
							testSample.at<float>(0, 2) = (float)circleRadius;
							int p = 3;
							for (int k = 0; k < circleCandidate.rows; k++)
							{
								for (int l = 0; l < circleCandidate.cols; l++)
								{
									testSample.at<float>(0, p) = (float)circleCandidate.at<uchar>(k, l) / 255;
									p++;
								}

							}
							
							annResult = ann->predict(testSample);

							if (annResult == 0)
							{
								circles.push_back(make_pair(circleRadius, circleCenter));
							}
						}
					}
				}

			}
		}
		singleContourImage = Mat(imgr2p.size(), imgr2p.type(), Scalar(0));
	}
	return circles;
}

//Lens Cleanign implemented based on Karols code
vector<Mat> LensCleaning()
{
	vector<Mat> returnVMat; //creating vector of Mat which will store images after lens cleaning
	if (images.size() > 30) //lens cleaning is performed only if there is sufficient number of images provided
	{
		auto begin = chrono::system_clock::now();
		Mat image = images[0]; //reading first image
		//Mat r2pImage = r2pTransform(image); //performing r2pow and log transform
		image.convertTo(image, CV_16UC1); //conversion to 16bit image
		Mat sumImage = Mat(image.size(), CV_16UC1, Scalar(0)); //alocating Mat object which will store sum of pixels
		//looping through rest of images to add all images to one
		for (int imageIterator = 0; imageIterator < images.size(); imageIterator++)
		{
			add(sumImage, image, sumImage); //adding images
			image = images[imageIterator];//reading next image
			image.convertTo(image, CV_16UC1); //convertion to 16bit image
		}
		
		Mat imageMean = Mat(image.size(), CV_16UC1, Scalar(0)); //alocating Mat object which will store average pixel value
		imageMean = sumImage / (images.size() - 4); //divding sum image by number of all images-> calculating mean of each pixel
		imageMean.convertTo(imageMean, CV_8UC1); //convertion back to 8bit image
		
		int avgGrayLevel = mean(imageMean)[0]; //average gray level value required to maitain the same gray level after noise removal
		//avgGrayLevel=sum(imageMean)[0]; //summing all pixels to one value
		//avgGrayLevel = avgGrayLevel / (image.rows*image.cols); //dividing sumed pixels values by number of pixels
		
		auto end = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = end - begin;
		for (int i = 0; i < images.size(); i++)
		{
			Mat out = images[i]; //reading first image
			for (int j = 0; j < out.rows; j++)
			{
				for (int l = 0; l < out.cols; l++)
				{
					out.at<uchar>(j, l) = out.at<uchar>(j, l) - imageMean.at<uchar>(j, l) + avgGrayLevel;
				}
			}
			returnVMat.push_back(out); //assigning cleaned image to return vector
		}
	}
	return returnVMat; //returning images in vector
}
vector<float> calcHist(Mat img) 
{
	vector<float> returnHist(256, 0);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			returnHist[img.at<uchar>(i, j)]++;
		}

	for (int i = 0; i < 256; i++)
	{
		returnHist[i] /= (float)439684;
	}

	return returnHist;
}
// reference source for function:
//http://answers.opencv.org/question/87394/calculating-the-distance-between-two-points/
float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

//this function classify image to certain class. in provided samples images with bubbles are sometimes very different and
//software uses histogram to classify image to proper class. Classification is performed based on all of provided images.
//Images will be classified to class which gets the majority of votes.
//using this method software can perform adaptive thresholding with properly adjusted thresholding parameters.
bool classifyImageType(string classifierName)
{
	Ptr<ml::ANN_MLP> annIC = ml::ANN_MLP::create();
	try
	{
		annIC = ml::ANN_MLP::load(classifierName);
	}
	catch (const std::exception& e)
	{
		cout << "Error while loading Neural Network for image classifier " << classifierName << endl;
		e.what();
		return false;
	}
	
	Mat rst = Mat(1, 5, CV_32FC1, Scalar(0));
	vector<int> classVote(5, 0);
	for (int i = 0; i < images.size(); i++)
	{
		vector<float> histC(256);
		Mat img = images[i];
		Mat imgr2p = r2pTransform(img);
		histC = calcHist(imgr2p);
		int p = annIC->predict(histC, rst);
		classVote[p]++;
		if (classVote[p] > images.size() / 2)
		{
			imgType = p + 1;
			return true;
		}
	}
	int maxTmp = 0;
	int maxIndx = 0;
	for (int i = 0; i < adaptiveThresholdParams.size(); i++)
	{
		if (classVote[i] > maxTmp)
		{
			maxTmp = classVote[i];
			maxIndx = i;
		}
	}
	imgType = maxIndx + 1;
	return true;
}
int main(int argc, const char **argv) {

	// Instructions for user
	//cout << "n - Next image (automatically saving image and all your changes" << endl;
	vector<int> saveParams;
	saveParams.push_back(IMWRITE_PNG_COMPRESSION);
	saveParams.push_back(9);
	//remove("outputFile.txt");
	calibration_factor = atof(argv[1]);  // calibration factor defined by user
	name_of_classifier = argv[3];        // name of classifier that should be used (for example Haar5.xml)
	number_of_drops = 0;                 // counter of droplets on all images
	for (int i = 4; i < argc; i++)
	{
		images.push_back(removeLeftBorder(imread(argv[i], IMREAD_GRAYSCALE)));
	}
	vector<Mat> imagesAfterLC;
	auto start = chrono::system_clock::now();
	
	if (!classifyImageType("ANN_ImageTypeClassifier.xml"))
	{
		return -1;
	}

	imagesAfterLC = LensCleaning();

	string edgeDropletClassifierName = "NN" + to_string(imgType) +".xml";
							
	try
	{
		ann = ann->load(edgeDropletClassifierName);
	}
	catch (const std::exception& e)
	{
		e.what();
		cout << "Failed to traing classifier using file " << edgeDropletClassifierName << endl;
		return -1;
	}

	path = argv[2];
	name_of_file = path + "outputFile.txt";       // creating file with diameters in pixel unit
	file.open(name_of_file, ios::trunc | ios::out);

	if (!file.is_open())
	{
		cout << "Failed to open file " << name_of_file << endl;
		return -1;
	}

	// Loop through all images in the input folder, it starts from 5 because path to images is 5th argument on the command window
	for (int k = 4; k < argc; k++)
	{
		path = argv[2];                                   // path to output folder
		
		keepProcessing = true;

		Mat imgLC;

		if (imagesAfterLC.size())
			imgLC = removeLeftBorder(imagesAfterLC[k - 4]);

		img = removeLeftBorder(images[k-4]);    // loading the image in grayscale (classifier was trained on greyscale images)
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

		//
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
			//tempPath = tempPath + "\\";
			tempPath = tempPath + name_of_image;
			file << "Name of image: "<<name_of_image<<endl;

			// iteration through all the vector with detected droplets
			for (int i = 0; i < drops.size(); i++) 
			{
				Rect r = drops[i];
				// calculating the center of droplet
				Point center(drops[i].x + drops[i].width*0.5, drops[i].y + drops[i].height*0.5 + 3);
				// calculating radius of droplets
				int radius = cvRound(abs(drops[i].width*0.5));
				int diameter = 2 * radius;

				Mat circlePart = ExtendRect(r, imgr2p,0.15);

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
					radius = newRadiusAvg;
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
					bubbles.push_back(make_pair( radius, Point(newCenter.x + (r.tl().x), newCenter.y + (r.tl().y))));

				}
				else
				{
					bubbles.push_back(make_pair(radius, center));
					centers.push_back(center);
					radiuses.push_back(radius);
					//circle(img1, center, radius, Scalar(58, 71, 244),2);
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
			
			edgeBubbles = findEdgeBubbles(NN, bubbles);

			vector<pair<float, Point>> bubblesFiltered;
			for (int g = 0; g < edgeBubbles.size(); g++)
			{
				bubbles.push_back(edgeBubbles[g]);
			}
			edgeBubbles = bubbles;
			vector<pair<float, Point>> edgeBubblesFiltered;
			vector<pair<float, Point>> bubblesTmp;
			pair<float,Point> b(0,Point(0,0));
			//filtering out similar bubbles (with similar center based on distance of centers of two bubbles and their radiuses).
			for (int i = 0; i < edgeBubbles.size(); i++)
			{
				float maxR = 0;
				int mIndx = 0;
				for (int m=0; m < edgeBubbles.size(); m++)
				{
					float dRatio = euclideanDist(edgeBubbles[i].second, edgeBubbles[m].second) / (edgeBubbles[i].first + edgeBubbles[m].first);
					if(m !=i && dRatio<=0.8)
					{
						bubblesTmp.push_back(edgeBubbles[m]);
					}
				}
				if (!bubblesTmp.size())
				{
					bubblesTmp.push_back(edgeBubbles[i]);
				}
				else
				{
					maxR = edgeBubbles[i].first;
					for (int l = 0; l < bubblesTmp.size(); l++)
					{
						if (bubblesTmp[l].first > maxR)
						{
							maxR = bubblesTmp[l].first;
							mIndx = l;
						}
					}
				}
				if (maxR != edgeBubbles[i].first&&  find(edgeBubblesFiltered.begin(), edgeBubblesFiltered.end(), (bubblesTmp[mIndx])) == edgeBubblesFiltered.end())
				{
					edgeBubblesFiltered.push_back(bubblesTmp[mIndx]);
					bubblesFiltered.push_back(bubblesTmp[mIndx]);
				}
				bubblesTmp.clear();
				b = make_pair(0, Point(0, 0));
			}
			for (int a = 0; a < bubblesFiltered.size(); a++)
			{
				circle(img1, bubblesFiltered[a].second, bubblesFiltered[a].first, Scalar(58, 71, 244), 2);
			}
			
			imwrite(tempPath, img1,saveParams);
			
			
			// Saving information from current image in files. Mean diameter, min, max and SMD
			
			file << "Nb : " << bubblesFiltered.size() << endl;
			double meanDiameter = 0;
			double maxDiameter = 0;
			double minDiameter=1000;
			vector<double> diamTmp;
			for (int m = 0; m < bubblesFiltered.size(); m++)
			{
				meanDiameter += bubblesFiltered[m].first * 2;
				if (maxDiameter < bubblesFiltered[m].first * 2)
				{
					maxDiameter = bubblesFiltered[m].first * 2;
				}
				if (minDiameter > bubblesFiltered[m].first * 2)
				{
					minDiameter = bubblesFiltered[m].first * 2;
				}
				diamTmp.push_back(calibration_factor * 2 * bubblesFiltered[m].first);
				diameters_in_mikrometers.push_back(calibration_factor * 2 * bubblesFiltered[m].first);
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

			number_of_drops = number_of_drops + bubblesFiltered.size();
			edgeBubblesFiltered.clear();
			bubblesFiltered.clear();
			bubblesTmp.clear();
		}
		// Cleaning all the vectors
		bubbles.clear();
		centers.clear();
		radiuses.clear();
		drops.clear();
	}

	// Calculating  Sauter mean diameter for all droplets on all images
	d3 = 0;
	d2 = 0;
	SMD = 0;

	double meanAll=0;
	double minAll=1000;
	double maxAll=0;

	for (int i = 0; i < diameters_in_mikrometers.size(); i++) {
		d3 = d3 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
		d2 = d2 + diameters_in_mikrometers[i] * diameters_in_mikrometers[i];
		meanAll += diameters_in_mikrometers[i];
		if (maxAll < diameters_in_mikrometers[i] )
		{
			maxAll = diameters_in_mikrometers[i] ;
		}
		if (minAll > diameters_in_mikrometers[i])
		{
			minAll = diameters_in_mikrometers[i];
		}
	}
	SMD = d3 / d2;
	auto end = chrono::system_clock::now();

	// Sauter Mean Diameter in file : "diametersInMikroMeters.txt"
	
	file << endl<< "NbAll : " << number_of_drops << endl;
	file << "SMDAll : " << SMD << endl;
	file << "MeanAll : " << (meanAll / diameters_in_mikrometers.size()) << endl;
	file << "MinAll : " << minAll << endl;
	file << "MaxAll : " << maxAll << endl;
	file << "Time : " << chrono::duration_cast<chrono::seconds>(end - start).count()<<" sec"<<endl;
	file.close();
	images.clear();
	imagesAfterLC.clear();
	radiuses.clear();
	adaptiveThresholdParams.clear();
	bubbles.clear();
	edgeBubbles.clear();
	cout << "Program finished without errors" << endl;
	return 0;
	
}