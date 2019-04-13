#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <chrono>
using namespace std;
using namespace cv;

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
Mat removeLeftBorder(Mat input)
{
	Mat tmp = Mat(input.size().height, input.size().width - 2, input.type());
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 2; j < input.cols; j++)
		{
			tmp.at<uchar>(i, j-2) = input.at<uchar>(i, j);
		}
	}
	return tmp;
}
Mat SpreadRect(Rect r,Mat sourceImage)
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

	Mat RectPart = sourceImage(Range(tl.y, br.y), Range(tl.x, br.x));
	return RectPart;
	
}
//This functions finds bubbles on edges and user classifies those bubbles to correct or not correct using key input
//if marked area is bubble on edge press enter to calssify it as bubble on edge
//if marked area is not bubble on edge press 'n' to classify it as not bubble on edge
void prepareOutputFile(int argc, char** argv)
{
	char key = 0;
	ofstream outputFile;
	outputFile.open("Out.txt");
	for (int argP = 1; argP < argc; argP++)
	{
		Mat img = imread(argv[argP], IMREAD_GRAYSCALE);
		Mat imgClean = imread(argv[argP], CV_LOAD_IMAGE_COLOR);
		Mat imgCleanCopy = imgClean.clone();
		Mat lbrImg = removeLeftBorder(img);

		Mat o = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat poly = o.clone();
		Point2f circleCenter;
		float circleRadius;

		Mat circleImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));

		Mat out;
		vector<vector<Point>> contours;
		vector<vector<Point>> circleContour;
		vector<Point> polygonContour;

		//some image preprocessing
		Mat r2p = r2pTransform(lbrImg);
		out = r2p.clone();
		threshold(out, out, 100, 255, THRESH_BINARY);
		Mat cannyOut;
		Canny(out, cannyOut, 50, 240, 7);
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
							approxPolyDP(contours[i], polygonContour, 0.1, true);
							minEnclosingCircle(polygonContour, circleCenter, circleRadius);

							circle(singleContourImage, circleCenter, circleRadius, Scalar(255));
							findContours(singleContourImage, circleContour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
							drawContours(imgClean, circleContour, -1, Scalar(0, 255, 0));

							Rect r = minAreaRect(circleContour[0]).boundingRect();

							Mat RectPart = SpreadRect(r, lbrImg);
							resize(RectPart, RectPart, Size(60, 60));
							rectangle(imgClean, r, Scalar(255, 0, 0));
							imshow("Image with Circle", imgClean);
							imshow("RectPart", RectPart);
							key = waitKey();
							outputFile << (int)circleCenter.x << " , " << (int)circleCenter.y << " , " << (int)circleRadius << " , ";
							for (int k = 0; k < RectPart.rows; k++)
							{
								for (int l = 0; l < RectPart.cols; l++)
								{
									outputFile << (int)RectPart.at<uchar>(k, l) << " , ";
								}

							}
							if (key == 13)
							{
								outputFile << "b" << endl;
							}
							if (key == 'n')
							{
								outputFile << "n" << endl;
							}
							break;
						}
					}
				}

			}
			imgClean = imgCleanCopy.clone();
			circleImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
			contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
			singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		}
	}
}

void testNeuralNetwork(int argc, char** argv, string annXMLName)
{
	float annResult = 0;
	int noOfFeatures = 3603;
	Mat trainData = Mat(232, noOfFeatures, CV_32FC1, Scalar(0));
	Mat trainClass = Mat(232, 2, CV_32FC1, Scalar(0));
	Mat result = Mat(1, 2, CV_32FC1, Scalar(0));

	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();

	auto start = chrono::steady_clock::now();
	ann = ml::ANN_MLP::load(annXMLName);
	auto end = chrono::steady_clock::now();

	if (!ann->isTrained())
	{
		cout << "Network not trained" << endl;
	}
	cout << "Loading NN from xml took: " << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << endl;
	for (int argP = 1; argP < argc; argP++)
	{
		Mat testSample = Mat(1, noOfFeatures, CV_32FC1, Scalar(0));
		Mat img = imread(argv[argP], IMREAD_GRAYSCALE);
		Mat imgClean = imread(argv[argP], CV_LOAD_IMAGE_COLOR);
		Mat imgCleanCopy = imgClean.clone();
		Mat lbrImg = removeLeftBorder(img);

		Mat o = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat poly = o.clone();
		Point2f circleCenter;
		float circleRadius;

		Mat circleImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		Mat singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));

		Mat out;
		vector<vector<Point>> contours;
		vector<vector<Point>> circleContour;
		vector<Point> polygonContour;

		//some image preprocessing
		Mat r2p = r2pTransform(lbrImg);
		out = r2p.clone();
		threshold(out, out, 100, 255, THRESH_BINARY);
		Mat cannyOut;
		Canny(out, cannyOut, 50, 240, 7);
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
							approxPolyDP(contours[i], polygonContour, 0.1, true);
							minEnclosingCircle(polygonContour, circleCenter, circleRadius);

							circle(singleContourImage, circleCenter, circleRadius, Scalar(255));
							findContours(singleContourImage, circleContour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
							drawContours(imgClean, circleContour, -1, Scalar(0, 255, 0));

							Rect r = minAreaRect(circleContour[0]).boundingRect();

							Mat RectPart = SpreadRect(r, lbrImg);
							resize(RectPart, RectPart, Size(60, 60));
							rectangle(imgClean, r, Scalar(255, 0, 0));
							imshow("Image with Circle", imgClean);
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
							auto start = chrono::steady_clock::now();
							annResult = ann->predict(testSample);
							auto end = chrono::steady_clock::now();
							
							cout << "Prediction time in seconds : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

							if (annResult == 0)
							{
								cout << "Bubble on edge" << endl;
							}
							waitKey();
							break;
						}
					}
				}

			}
			imgClean = imgCleanCopy.clone();
			circleImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
			contImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
			singleContourImage = Mat(lbrImg.size(), lbrImg.type(), Scalar(0));
		}
	}
}
void trainNeuralNetwork()
{
	ifstream inputFile;
	inputFile.open("Out.txt");
	if (!inputFile.is_open())
	{
		cout << "Cannot open file Out.txt" << endl;
	}
	string line, lineTmp, fileName;
	int noOfFeatures = 3603;
	Mat trainData = Mat(232, noOfFeatures, CV_32FC1, Scalar(0));
	Mat trainClass = Mat(232, 2, CV_32FC1, Scalar(0));
	int trainDataRowPointer = 0;

	while (trainDataRowPointer < 232)
	{
		int i = 0;
		for (i; i < 3603; i++)
		{
			inputFile >> line;
			inputFile >> lineTmp;
			if (i >= 3)
			{
				trainData.at<float>(trainDataRowPointer, i) = (float)stof(line) / 255;
			}
			else
			{
				trainData.at<float>(trainDataRowPointer, i) = (float)stof(line);
			}
		}
		inputFile >> line;
		if (line == "b")
		{
			trainClass.at<float>(trainDataRowPointer, 0) = 1.0;
		}
		else
		{
			trainClass.at<float>(trainDataRowPointer, 1) = 1.0;
		}
		trainDataRowPointer++;
	}

	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	float lay[3] = { noOfFeatures,100,2 };
	Mat layM = Mat(3, 1, CV_32F, lay);
	ann->setLayerSizes(layM);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0.8, 0.8);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.1, 0.1);

	cout << "Training started..." << endl;
	auto start = chrono::steady_clock::now();
	ann->train(trainData, ml::ROW_SAMPLE, trainClass);
	auto end = chrono::steady_clock::now();

	cout << "Training time in seconds : " << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << endl;

	if (!ann->isTrained())
	{
		cout << "Network not trained" << endl;
	}

	cout << "Network trained, saving network to ANN.xml" << endl;

	ann->save("ANN.xml");
}
//int main(int argc, char** argv)
//{
//	prepareOutputFile(argc, argv);
//}

