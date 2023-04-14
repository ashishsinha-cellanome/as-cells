// Yolov5ObjectDetector.h
#ifndef YOLOV5OBJECTDETECTOR_H
#define YOLOV5OBJECTDETECTOR_H

#include <opencv2/opencv.hpp>

// Namespaces
using namespace std;

class Yolov5ObjectDetector
{
private:
    cv::dnn::Net _net;
    vector<string> _classNames;
    string _weightsPath;
    string _namesPath;
    float _modelInputWidth;
    float _modelInputHeight;
    float _confidence;
    float _nmsThreshold;
    string _device;
    
    void _postProcess(cv::Mat& inputImage, vector<cv::Mat>& modelPreds, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores);
    void _postProcess(cv::Mat& inputImage, vector<cv::Mat>& modelPreds, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage);

    
public:
    //Default Constructor
    Yolov5ObjectDetector();
    //Parameterized Constructor
    Yolov5ObjectDetector(bool useGpu);
    Yolov5ObjectDetector(string weightsPath, string namesPath, float modelInputWidth, float modelInputHeight, float confidence, float nmsThreshold, bool useGpu);
    void loadWeights();
    float getInputWidth();
    float getInputHeight();
    vector<cv::Mat> runModel(cv::Mat& inputImage);
    bool detect(cv::Mat& inputImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores);
    bool detect(cv::Mat& inputImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage);
    bool detectByCropping(cv::Mat& inputImage, const vector<cv::Rect>& cropCorners, float nmsThresholdForRemovingDuplicates, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores);
    bool detectByCropping(cv::Mat& inputImage, const vector<cv::Rect>& cropCorners, float nmsThresholdForRemovingDuplicates, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage);
};

#endif
