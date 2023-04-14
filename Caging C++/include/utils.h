// Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

// Namespaces
using namespace std;

float calculateIou(const cv::Rect& box1, const cv::Rect& box2);
float boxArea(const cv::Rect& box);
vector<vector<float>> iouBatch(const vector<cv::Rect>& boxSet1, const vector<cv::Rect>& boxSet2);
void addDetection(cv::Mat& debugImage, string label, int left, int top);
void displayResults(cv::Mat& debugImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, vector<string> classIdToClassNameMap);
void addRuntime(cv::Mat& debugImage, string runtimeString);
#endif
