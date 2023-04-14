#include <opencv2/opencv.hpp>

// Namespaces
using namespace std;

// Text parameters (for plotting the results)
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors (for plotting the results)
cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);

// IoU calculation
float calculateIou(const cv::Rect& box1, const cv::Rect& box2) 
{


    float w = max(min((box1.x + box1.width), (box2.x + box2.width)) - max(box1.x, box2.x), 0);
    float h = max(min((box1.y + box1.height), (box2.y + box2.height)) - max(box1.y, box2.y), 0);

    if (w * h == 0) {
        return 0;
    }

    return w * h / (box1.width * box1.height + box2.width * box2.height - w * h);

}

float boxArea(const cv::Rect& box) {
    return (float) box.height * box.width;
}

// Batch IoU between two groups of boxes (needed for combining detections between crops)
vector<vector<float>> iouBatch(const vector<cv::Rect>& boxSet1, const vector<cv::Rect>& boxSet2) 
{

    vector<vector<float>> iouMatrix;

    for (int i = 0; i < boxSet1.size(); i++)
    {
        vector<float> iouValues;
        for (int j = 0; j < boxSet2.size(); j++)
        {
            float iou = calculateIou(boxSet1[i], boxSet2[j]);
            iouValues.push_back(iou);

        }
        iouMatrix.push_back(iouValues);
    }

    return iouMatrix;
}

// Draw the predicted bounding box
void addDetection(cv::Mat& debugImage, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, labelSize.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + labelSize.width, top + labelSize.height + baseLine);
    // Draw black rectangle.
    cv::rectangle(debugImage, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(debugImage, label, cv::Point(left, top + labelSize.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

void displayResults(cv::Mat& debugImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, vector<string> classIdToClassNameMap)
{
    for (int i = 0; i < outBoxes.size(); i++)
    {
        cv::Rect box = outBoxes[i];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box
        cv::rectangle(debugImage, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // Get the label for the class name and its confidence
        string label = cv::format("%.2f", outScores[i]);
        label = classIdToClassNameMap[outLabels[i]] + ":" + label;
        // Draw class labels
        addDetection(debugImage, label, left, top);
    }
    return;
}

void addRuntime(cv::Mat& debugImage, string runtimeString)
{
    cv::putText(debugImage, runtimeString, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    return;
}
