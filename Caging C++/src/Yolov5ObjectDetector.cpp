// Yolov5ObjectDetector.cpp
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "Yolov5ObjectDetector.h"
#include "utils.h"


// Namespaces
using namespace std;

// Constants
const string MODEL_WEIGHTS_PATH = "weights/batch_1_batch_2_batch_3_images_20_epochs.onnx";
const string CLASS_NAMES_PATH = "weights/cells.names";
const float SCORE_THRESHOLD = 0.5; 
const float DEFAULT_DETECTION_CONFIDENCE = 0.4;
const float DEFAULT_NMS_THRESHOLD = 0.3;
const float INPUT_IMAGE_WIDTH = 640.0; // YOLOv5 model input width
const float INPUT_IMAGE_HEIGHT = 640.0; // YOLOv5 model input height

float Yolov5ObjectDetector::getInputWidth()
{
    return _modelInputWidth;
}
float Yolov5ObjectDetector::getInputHeight()
{
    return _modelInputHeight;
}

void Yolov5ObjectDetector::_postProcess(cv::Mat& inputImage, vector<cv::Mat>& modelPreds, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores)
{
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<int> labels;
    vector<float> scores;
    vector<cv::Rect> boxes;

    // Resizing factor.
    float x_factor = inputImage.cols / _modelInputWidth;
    float y_factor = inputImage.rows / _modelInputHeight;

    float* data = (float*)modelPreds[0].data;

    // modelPreds is a vector of num_images OpenCV Mat arrays, each of size 25200 x (5 + num_classes)
    // Here, we are passing only one image to the model, hence only has size of 1 and modelPreds[0] is of interest
    // 25200 is fixed and is the number of detection proposals returned by the model (tied to the number of anchors, which is fixed)
    // the first 4 element is each row are the coordinates of the bounding box center_x, center_y, w, h
    // the 5th element is the detection score
    // the 6th element and after are the likelihood for each class
    const int dimensions = 5 + _classNames.size();
    const int numDetections = 25200; // this is a fixed number returned by the model
    // Iterate through 25200 detections.
    for (int i = 0; i < numDetections; i++)
    {
        float score = data[4];
        // Discard bad detections and continue.
        if (score >= min(SCORE_THRESHOLD, _confidence))
        {
            float* classesScoresFloat = data + 5;
            // Create a 1 x num_classes Mat and store class scores of the classes
            // we do this to use OpenCV implementation of minMaxLoc, which can return the maximum value and index of an array
            cv::Mat classesScores(1, _classNames.size(), CV_32FC1, classesScoresFloat);
            // Perform minMaxLoc and acquire index of best class score
            cv::Point classId;
            double maxClassScore;
            minMaxLoc(classesScores, 0, &maxClassScore, 0, &classId);
            // Continue if the class score is above the threshold
            if (maxClassScore >= _confidence)
            {
                // Store class ID and confidence in the pre-defined respective vectors
                scores.push_back(maxClassScore);
                // classId is returned as a Point object, since this was a 1-D array, taking x is enough
                labels.push_back(classId.x);

                // Center
                float cx = data[0];
                float cy = data[1];
                // Box dimension
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += dimensions;
    }

    // Perform Non Maximum Suppression per class and retuned the results in the passed pointers

    // repeat for all label IDs
    for (int i = 0; i < _classNames.size(); i++)
    {
        vector<int> indices;
        vector<cv::Rect> boxesThisClass;
        vector<int> labelsThisClass;
        vector<float> scoresThisClass;

        for (int j = 0; j < boxes.size(); j++)
        {
            if (labels[j] == i)
            {
                boxesThisClass.push_back(boxes[j]);
                labelsThisClass.push_back(labels[j]);
                scoresThisClass.push_back(scores[j]);
            }
        }
        
        cv::dnn::NMSBoxes(boxesThisClass, scoresThisClass, _confidence, _nmsThreshold, indices);
        for (int j = 0; j < indices.size(); j++)
        {
            int idx = indices[j];
            outBoxes.push_back(boxesThisClass[idx]);
            outLabels.push_back(labelsThisClass[idx]);
            outScores.push_back(scoresThisClass[idx]);
        }
    }
    return;
}
 
void Yolov5ObjectDetector::_postProcess(cv::Mat& inputImage, vector<cv::Mat>& modelPreds, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage)
{
 
    _postProcess(inputImage, modelPreds, outBoxes, outLabels, outScores);
    
    // make a copy of the input image
    debugImage = inputImage.clone();
    // Add model runtime
    vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = _net.getPerfProfile(layersTimes) / freq;
    string runtimeString = cv::format("Inference time : %.2f ms", t);
    // display all detections with runtime
    displayResults(debugImage, outBoxes, outLabels, outScores, _classNames);
    addRuntime(debugImage, runtimeString);
    return;
    
}

//Default Constructor
Yolov5ObjectDetector::Yolov5ObjectDetector(): _weightsPath(MODEL_WEIGHTS_PATH), _namesPath(CLASS_NAMES_PATH), _modelInputWidth(INPUT_IMAGE_WIDTH), 
_modelInputHeight(INPUT_IMAGE_HEIGHT), _confidence(DEFAULT_DETECTION_CONFIDENCE), _nmsThreshold(DEFAULT_NMS_THRESHOLD), _device("cpu")
{
    loadWeights();
}

Yolov5ObjectDetector::Yolov5ObjectDetector(bool useGpu): _weightsPath(MODEL_WEIGHTS_PATH), _namesPath(CLASS_NAMES_PATH), _modelInputWidth(INPUT_IMAGE_WIDTH), 
_modelInputHeight(INPUT_IMAGE_HEIGHT), _confidence(DEFAULT_DETECTION_CONFIDENCE), _nmsThreshold(DEFAULT_NMS_THRESHOLD), _device("cpu")
{
    if (useGpu) {
        _device = "gpu";
    }
    loadWeights();
}

//Parameterized Constructor
Yolov5ObjectDetector::Yolov5ObjectDetector(string weightsPath, string namesPath, float modelInputWidth, float modelInputHeight, float confidence, float nmsThreshold, bool useGpu):  _weightsPath(weightsPath), _namesPath(namesPath), _modelInputWidth(modelInputWidth), 
_modelInputHeight(modelInputHeight), _confidence(confidence), _nmsThreshold(nmsThreshold), _device("cpu")
{
    if (useGpu) {
        _device = "gpu";
    }
    loadWeights();
}

void Yolov5ObjectDetector::loadWeights() {

    // Load class names
    ifstream ifs(_namesPath);
    string line;

    while (getline(ifs, line))
    {
        _classNames.push_back(line);
    }
    
    for (int i = 0; i < _classNames.size(); i++)
    {
        cout << "class name for class ID " << i << ": " << _classNames[i] << endl;
    }

    // Load the ONNX model
    _net = cv::dnn::readNet(_weightsPath);
    if (_device == "cpu")
    {
        cout << "Using CPU device" << endl;
        _net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    }
    else if (_device == "gpu")
    {
        cout << "Using GPU device" << endl;
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
         // _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
}

vector<cv::Mat> Yolov5ObjectDetector::runModel(cv::Mat& inputImage)
{
    // Convert the input image to a blob, scale 
    cv::Mat blob;
    // the last 3 arguments are the image mean = (0, 0, 0), swapRB = True, crop = False
    cv::dnn::blobFromImage(inputImage, blob, 1. / 255., cv::Size(_modelInputWidth, _modelInputHeight), cv::Scalar(), true, false);

    _net.setInput(blob);

    // Forward propagate
    vector<cv::Mat> detections;
    _net.forward(detections, _net.getUnconnectedOutLayersNames());
    return detections;
}

bool Yolov5ObjectDetector::detect(cv::Mat& inputImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores) {
    // Check if the aspect ratio of the input image is almost the same as the aspect ratio of the model input size
    // if not, then the input image will be resized without keeping its aspect ratio when it 
    // is passed to the model, and this may lead to inaccurate detection
    float aspectRatioDiff = (inputImage.cols * _modelInputHeight) / (inputImage.rows * _modelInputWidth) - 1;
    if (abs(aspectRatioDiff) > 0.1) {
        string warning = cv::format("The input image has a different aspect ratio: %.2f that the model! The results may not be accurate", inputImage.cols / (1.0 * inputImage.rows));
        cout << warning << endl;
        return false;
    }
    
    // Forward propagate
    vector<cv::Mat> detections = runModel(inputImage);
    _postProcess(inputImage, detections, outBoxes, outLabels, outScores);

    return true;
}

bool Yolov5ObjectDetector::detect(cv::Mat& inputImage, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage) {
    // Check if the aspect ratio of the input image is almost the same as the aspect ratio of the model input size
    // if not, then the input image will be resized without keeping its aspect ratio when it 
    // is passed to the model, and this may lead to inaccurate detection
    float aspectRatioDiff = (inputImage.cols * _modelInputHeight) / (inputImage.rows * _modelInputWidth) - 1;
    if (abs(aspectRatioDiff) > 0.1) {
        string warning = cv::format("The input image has a different aspect ratio: %.2f that the model! The results may not be accurate", inputImage.cols / (1.0 * inputImage.rows));
        cout << warning << endl;
        return false;
    }

    // Forward propagate
    vector<cv::Mat> detections = runModel(inputImage);
    _postProcess(inputImage, detections, outBoxes, outLabels, outScores, debugImage);

    return true;
}

bool Yolov5ObjectDetector::detectByCropping(cv::Mat& inputImage, const vector<cv::Rect>& cropCorners, float nmsThresholdForRemovingDuplicates, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores)
{
    if (cropCorners.size() == 0) 
    {
        cout << "No crop corners are provided for running YOLOv5 model on sub-images. Returning no detections!" << endl;
        return false;
    }
 
    int H = inputImage.rows;
    int W = inputImage.cols;

    // Check if all the crop sub-images are of the same size,
    // if not, make them equal size
    int cropWidth = 0;
    int cropHeight = 0;

    for (int i = 0; i < cropCorners.size(); i++) {
        cv::Rect corners;
        corners = cropCorners[i];
        int width;
        int height;
        width = (int) min(corners.x + corners.width, W) - max(corners.x, 0);
        height = (int) min(corners.y + corners.height, H) - max(corners.y, 0);
        if (width < 0 || height < 0)
        {
            cout << "Incorrect corners are provided for running YOLOv5 model on sub-images. Returning no detections!" << endl;
            return false;
        }
        cropWidth = max(cropWidth, width);
        cropHeight = max(cropHeight, height);

    }
    
    // List to contain intermediate results, each element in the list is the results of one of the crops
    vector<vector<cv::Rect>> boxes;
    vector<vector<int>> labels;
    vector<vector<float>> scores;


    // Combine the results, filter them based on the score, and update the coordinates of the bounding boxes for applying NMS later
    // A list to keep track of cropped sub-images with at least one object detection
    vector<int>  cropIdsWithDetection;
    for (int i = 0; i < cropCorners.size(); i++) {
        cv::Rect corners;
        corners = cropCorners[i];
        //  Enlarge the crop if necessary to make all the same size
        cv::Rect roi = cv::Rect(corners.x, corners.y, cropWidth, cropHeight);
        // Crop the image and run the model
        cv::Mat croppedImage = inputImage(roi);
        // Outputs
        vector<cv::Rect> outBoxes;
        vector<int> outLabels;
        vector<float> outScores;
        bool success = detect(croppedImage, outBoxes, outLabels, outScores);

        if (!success || outScores.size() == 0)
        {
            continue;
        }

        // Find the bounding boxes close to the boundaries of the cropped image
        // these boxes most probably are truncated (because they are close to the boundary)
        // for a properly designed image crops, the overlapping section(in x or y dimensions)
        // between two adjacent crops is larger than than the largest object (in each dimension)
        // hence an object can only cross one boundary of an overlapping part and will
        // definitely lie completely in another cropped image
        // modify the score for these detected boxes (assign the minimum score of self._confidence)
        // to give them lower priority during NMS when the results in the overlapping parts
        // of cropped images are combined

        for (int j = 0; j < outBoxes.size(); j++)
        {
            if (outBoxes[j].x < 4 || outBoxes[j].y < 4 || (outBoxes[j].x + outBoxes[j].width) > (cropWidth - 4) || (outBoxes[j].y + outBoxes[j].height) >(cropHeight - 4))
            {
                outScores[j] = _confidence;
            }

            // Now update/shift the boxes to the original image cooridnate
            outBoxes[j].x += corners.x;
            outBoxes[j].y += corners.y;

        }
        cropIdsWithDetection.push_back(i);
        boxes.push_back(outBoxes);
        labels.push_back(outLabels);
        scores.push_back(outScores);
    }

    // No object detected, return
    if (cropIdsWithDetection.size() == 0)
    {
        return true;
    }
    

    // Now compare the detection results of one crop with the detections in the
    // rest of the image to identify objects that are uniquely detected in the
    // crop and should be kept
    // this part is needed to pick one object in the overlapping crop areas when
    // detected by the detector in multiple crops
    // note that we need to compare the detections in one crop only with overlapping
    // crops; but here we are doing it for all for simplicity of implementation
    // TODO: improve it in the future by passing the indexes of the neighboring crops

    // To decide which object to keep when detected in multiple crops, we compute
    // the IoU (for objects of the same class) between the crop under consideration and the rest
    // then we keep objects in the crop under consideration that have
    // IoU <= nms_threshold_for_combining_crop_results with objects detected in the rest
    // we also keep objects with IoU > nmsThresholdForRemovingDuplicates, if the detection
    // score for the object in the crop is higher than the objects in the other crops
    
    for (int idx = 0; idx < cropIdsWithDetection.size(); idx++)
    {
        vector<int> cropLabels = labels[idx];
        vector<float> cropScores = scores[idx];
        vector<cv::Rect> cropBoxes = boxes[idx];

        // Number of detections on other crops
        int numDetectionsInRest = 0;
        for (int i = 0; i < cropIdsWithDetection.size(); i++)
        {
            if (i != idx)
            {
                numDetectionsInRest += labels[i].size();
            }
        }

        if (numDetectionsInRest == 0)
        {
            // Keep all detections in this crop and go to the next crop (we can exit as well
            for (int i = 0; i < cropLabels.size(); i++)
            {
                outBoxes.push_back(cropBoxes[i]);
                outLabels.push_back(cropLabels[i]);
                outScores.push_back(cropScores[i]);
            }
            continue;
        }
        // Detection labels for detections in other cropped sub-images
        vector<int> restLabels;
        // Detection scores for detections in other cropped sub-images
        vector<float> restScores;
        // Detection boxes for detections in the other cropped sub-images
        vector<cv::Rect> restBoxes;
        for (int i = 0; i < cropIdsWithDetection.size(); i++)
        {
            if (i != idx)
            {   
                for (int j = 0; j < labels[i].size() ; j++)
                {
                    restLabels.push_back(labels[i][j]);
                    restScores.push_back(scores[i][j]);
                    restBoxes.push_back(boxes[i][j]);
                }
            }
        }

        // repeat for all label IDs
        for (int labelId = 0; labelId < _classNames.size(); labelId++)
        {
            // the indexes of detections of the same label in each crop and rest set
            vector<int> cropClassIdxs;
            vector<int> restClassIdxs;

            for (int i = 0; i < cropLabels.size(); i++)
            {
                if (cropLabels[i] == labelId)
                {
                    cropClassIdxs.push_back(i);
                }
            }
            for (int i = 0; i < restLabels.size(); i++)
            {
                if (restLabels[i] == labelId)
                {
                    restClassIdxs.push_back(i);
                }
            }

            if (cropClassIdxs.size() == 0)
            {
                continue;
            }

            // Indexes of detections (with respect to filtered cropClassIdxs to keep from this detection
            vector<int> idxsToKeep;
            
            vector <vector<float>> iouMatrix;

            if (restClassIdxs.size() == 0)
            {
                for (int i = 0; i < cropClassIdxs.size(); i++)
                {
                    idxsToKeep.push_back(i);
                }

            }
            else
            {
                //  Compute the IoU matrix betwen the detection boxes of the same label between this crop and the rest
                vector<cv::Rect> cropBoxesOfThisLabel;
                vector<cv::Rect> restBoxesOfThisLabel;
                for (int i = 0; i < cropClassIdxs.size(); i++)
                {
                    cropBoxesOfThisLabel.push_back(cropBoxes[cropClassIdxs[i]]);
                }
                for (int i = 0; i < restClassIdxs.size(); i++)
                {
                    restBoxesOfThisLabel.push_back(restBoxes[restClassIdxs[i]]);
                }

                iouMatrix = iouBatch(cropBoxesOfThisLabel, restBoxesOfThisLabel);

                // Keep the bounding boxes from the crop that have
                // 1) IoU <= nmsThresholdForRemovingDuplicates with the boxes in the rest
                //    of the crops, or
                // 2) IoU > nmsThresholdForRemovingDuplicates and the detection score for
                //    the box in the crop is higher than the detection scores for the boxes in the rest

                // First condition:  IoU <= nmsThresholdForRemovingDuplicates
                for (int i = 0; i < iouMatrix.size(); i++)
                {
                    float maxIou = 0;
                    for (int j = 0; j < iouMatrix[i].size(); j++)
                    {
                        if (iouMatrix[i][j] > maxIou)
                        {
                            maxIou = iouMatrix[i][j];
                        }
                    }

                    if (maxIou < nmsThresholdForRemovingDuplicates)
                    {
                        idxsToKeep.push_back(i);
                    }

                }
            }
            for (int i = 0; i < cropClassIdxs.size(); i++)
            {
                // Detection score of this object in the crop under consideration
                float cropDetScore = cropScores[cropClassIdxs[i]];
                // A flag to indicate this detection has already been decided to keep
                bool shouldKeepThisDetFromCrop = false;
                for (int j = 0; j < idxsToKeep.size(); j++)
                {
                    if (j == i)
                    {
                        shouldKeepThisDetFromCrop = true;
                        break;
                    }
                }

                if (shouldKeepThisDetFromCrop)
                {
                    // We need to report this detection
                    outBoxes.push_back(cropBoxes[cropClassIdxs[i]]);
                    outLabels.push_back(cropLabels[cropClassIdxs[i]]);
                    outScores.push_back(cropScores[cropClassIdxs[i]]);
                }
                else
                {
                    // There are some boxes in the rest of the crops with IoU more then the threshold
                    // find the maximum detection scores among the objects with IoU more than the threshold
                    // also find the area of the bounding box for that detection (needed to break a tie in case
                    // scores are equal)
                    // The score of the detection from the rest of the crops that has the maximum detection score and also
                    // IoU greater than the threshold 
                    float restDetScore = 0;
                    // Index of that box (with respect to restClassIdxs)
                    int maxIndex = -1;
                    for (int j = 0; j < iouMatrix[i].size(); j++)
                    {
                        if (iouMatrix[i][j] > nmsThresholdForRemovingDuplicates && restScores[restClassIdxs[j]] > restDetScore)
                        {
                            restDetScore = restScores[restClassIdxs[j]];
                            maxIndex = j;

                        }
                    }
                    // Areas of the matching boxes
                    float cropBoxArea = boxArea(cropBoxes[cropClassIdxs[i]]);
                    float restBoxArea = boxArea(restBoxes[restClassIdxs[i]]);
                    // In case of a tie, pick the box with ther larger area
                    // a tie can happen specially when the objects are both near a common boundary (e.g., a side of
                    // the image) of the crops (we reduce the scores for both to the threshold score and they become
                    // equal)
                    if (cropDetScore > restDetScore || (cropDetScore == restDetScore && cropBoxArea >= restBoxArea))
                    {
                        // Keep this object as it has the highest score among all
                        outBoxes.push_back(cropBoxes[cropClassIdxs[i]]);
                        outLabels.push_back(cropLabels[cropClassIdxs[i]]);
                        outScores.push_back(cropDetScore);
                        
                        if (cropDetScore == restDetScore)
                        {
                            // This is added to break the tie if both areas and scores are equal
                            // so we will not add the same box twice when considering it
                            // in another crop
                            scores[idx][cropClassIdxs[i]] += 1e-5;
                        }
                    }

                }

            }

        }

    }
    return true;
}

bool Yolov5ObjectDetector::detectByCropping(cv::Mat& inputImage, const vector<cv::Rect>& cropCorners, float nmsThresholdForRemovingDuplicates, vector<cv::Rect>& outBoxes, vector<int>& outLabels, vector<float>& outScores, cv::Mat& debugImage)
{
    bool success = detectByCropping(inputImage, cropCorners, nmsThresholdForRemovingDuplicates, outBoxes, outLabels, outScores);
    if (!success) 
    {
        return false;
    }
    // make a copy of the input image
    debugImage = inputImage.clone();
    // display all detections
    displayResults(debugImage, outBoxes, outLabels, outScores, _classNames);
    return success;
}
