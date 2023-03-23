// Include Libraries
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

// Namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants
const string MODEL_WEIGHTS_PATH = "weights/microscope-images-batch-1-091922-not-reviewed_20_epochs.onnx";
const string CLASS_NAMES_PATH = "weights/cells.names";
const float SCORE_THRESHOLD = 0.5; 
const float DEFAULT_DETECTION_CONFIDENCE = 0.4;
const float DEFAULT_NMS_THRESHOLD = 0.3;
const float INPUT_IMAGE_WIDTH = 640.0; // YOLOv5 model input width
const float INPUT_IMAGE_HEIGHT = 640.0; // YOLOv5 model input height
const vector<int> IMAGE_RESOLUTION_IX81 = { 2000, 1600 }; // Image resolution for ix-81 microscope
const vector<int> IMAGE_RESOLUTION_BB2 = { 4512, 4512 }; // Image resolution for Breadboard (BB2)
const vector<int> RESIZED_IMAGE_IX81 = { 1000, 800 }; // // The size of resized image for ix-81 images before cropping and running the model on the crops
const vector<int> RESIZED_IMAGE_BB2 = { 2880, 2880 }; // The size of resized image for Breadboard images before cropping and running the model on the crops
// Crop sizes for ix-81
const vector<Rect> CROP_CORNERS_IX81 = { Rect(0, 0, 640, 640 ), Rect(0, 160, 640, 640), Rect(360, 0, 640, 640), Rect(360, 160, 640, 640) };
// Crop sizes for Breadboard (BB2)
const vector<Rect> CROP_CORNERS_BB2 = {
    Rect(0, 0, 640, 640),    Rect(0, 560, 640, 640),    Rect(0, 1120, 640, 640),    Rect(0, 1680, 640, 640),    Rect(0, 2240, 640, 640),
    Rect(560, 0, 640, 640),  Rect(560, 560, 640, 640),  Rect(560, 1120, 640, 640),  Rect(560, 1680, 640, 640),  Rect(560, 2240, 640, 640),
    Rect(1120, 0, 640, 640), Rect(1120, 560, 640, 640), Rect(1120, 1120, 640, 640), Rect(1120, 1680, 640, 640), Rect(1120, 2240, 640, 640),
    Rect(1680, 0, 640, 640), Rect(1680, 560, 640, 640), Rect(1680, 1120, 640, 640), Rect(1680, 1680, 640, 640), Rect(1680, 2240, 640, 640),
    Rect(2240, 0, 640, 640), Rect(2240, 560, 640, 640), Rect(2240, 1120, 640, 640), Rect(2240, 1680, 640, 640), Rect(2240, 2240, 640, 640) 
};
// Bit depth for both ix-81 and Breadboard images
const int BIT_DEPTH = 14;


// Text parameters (for plotting the results)
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors (for plotting the results)
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

// IoU calculation
float calculateIou(const Rect& box1, const Rect& box2) 
{


    float w = max(min((box1.x + box1.width), (box2.x + box2.width)) - max(box1.x, box2.x), 0);
    float h = max(min((box1.y + box1.height), (box2.y + box2.height)) - max(box1.y, box2.y), 0);

    if (w * h == 0) {
        return 0;
    }

    return w * h / (box1.width * box1.height + box2.width * box2.height - w * h);

}

float boxArea(const Rect& box) {
    return (float) box.height * box.width;
}

// Batch IoU between two groups of boxes (needed for combining detections between crops)
vector<vector<float>> iouBatch(const vector<Rect>& boxSet1, const vector<Rect>& boxSet2) 
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
void addDetection(Mat& debugImage, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size labelSize = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, labelSize.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + labelSize.width, top + labelSize.height + baseLine);
    // Draw black rectangle.
    rectangle(debugImage, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(debugImage, label, Point(left, top + labelSize.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

class ModelDetections
{
public:
    vector<Rect> boxes;
    vector<int> labels;
    vector<float> scores;
    Mat debugImage;

    // Default constructor
    ModelDetections(){}

    // Constructor
    ModelDetections(vector<Rect>& detectionBoxes, vector<int>& detectionLabels, vector<float>& detectionScores)
    {
        boxes = detectionBoxes;
        labels = detectionLabels;
        scores = detectionScores;
    }
    ModelDetections(vector<Rect>& detectionBoxes, vector<int>& detectionLabels, vector<float>& detectionScores, Mat& image)
    {
        boxes = detectionBoxes;
        labels = detectionLabels;
        scores = detectionScores;
        debugImage = image;
    }
};

class Yolov5ObjectDetector
{
private:
    Net _net;
    vector<string> _classNames;
    string _weightsPath;
    string _namesPath;
    float _modelInputWidth;
    float _modelInputHeight;
    float _confidence;
    float _nmsThreshold;
    string _device;

    ModelDetections _postProcess(Mat& inputImage, vector<Mat>& modelPreds, bool displayResults)
    {
        // Initialize vectors to hold respective outputs while unwrapping detections.
        vector<int> labels;
        vector<float> scores;
        vector<Rect> boxes;

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
                Mat classesScores(1, _classNames.size(), CV_32FC1, classesScoresFloat);
                // Perform minMaxLoc and acquire index of best class score
                Point classId;
                double maxClassScore;
                minMaxLoc(classesScores, 0, &maxClassScore, 0, &classId);
                // Continue if the class score is above the threshold.
                if (maxClassScore >= _confidence)
                {
                    // Store class ID and confidence in the pre-defined respective vectors.

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
                    boxes.push_back(Rect(left, top, width, height));
                }

            }
            // Jump to the next column.
            data += dimensions;
        }

        // Perform Non Maximum Suppression per class
        vector<Rect> outBoxes;
        vector<int> outLabels;
        vector<float> outScores;

        // repeat for all label IDs
        for (int i = 0; i < _classNames.size(); i++)
        {
            vector<int> indices;
            vector<Rect> boxesThisClass;
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
            
            NMSBoxes(boxesThisClass, scoresThisClass, _confidence, _nmsThreshold, indices);
            for (int j = 0; j < indices.size(); j++)
            {
                int idx = indices[j];
                outBoxes.push_back(boxesThisClass[idx]);
                outLabels.push_back(labelsThisClass[idx]);
                outScores.push_back(scoresThisClass[idx]);
            }
        }
        
        ModelDetections outputs;
        
        if (displayResults)
        {
            // make a copy of the input image
            Mat debugImage = inputImage.clone();
            for (int i = 0; i < outBoxes.size(); i++)
            {
                Rect box = outBoxes[i];

                int left = box.x;
                int top = box.y;
                int width = box.width;
                int height = box.height;
                // Draw bounding box.
                rectangle(debugImage, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

                // Get the label for the class name and its confidence.
                string label = format("%.2f", outScores[i]);
                label = _classNames[outLabels[i]] + ":" + label;
                // Draw class labels.
                addDetection(debugImage, label, left, top);
            }
            // Add model runtime
            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = _net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time : %.2f ms", t);
            putText(debugImage, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
            outputs = ModelDetections(outBoxes, outLabels, outScores, debugImage);
        }
        else
        {
            outputs = ModelDetections(outBoxes, outLabels, outScores);
        }

        return outputs;
    }


    
public:
    //Default Constructor
    Yolov5ObjectDetector()
    {
        _weightsPath = MODEL_WEIGHTS_PATH;
        _namesPath = CLASS_NAMES_PATH;
        _modelInputWidth = INPUT_IMAGE_WIDTH;
        _modelInputHeight = INPUT_IMAGE_HEIGHT;
        _confidence = DEFAULT_DETECTION_CONFIDENCE;
        _nmsThreshold = DEFAULT_NMS_THRESHOLD;
        _device = "cpu";
        loadWeights();
    }

    Yolov5ObjectDetector(bool useGpu)
    {
        _weightsPath = MODEL_WEIGHTS_PATH;
        _namesPath = CLASS_NAMES_PATH;
        _modelInputWidth = INPUT_IMAGE_WIDTH;
        _modelInputHeight = INPUT_IMAGE_HEIGHT;
        _confidence = DEFAULT_DETECTION_CONFIDENCE;
        _nmsThreshold = DEFAULT_NMS_THRESHOLD;
        _device = "cpu";
        if (useGpu) {
            _device = "gpu";
        }
        loadWeights();
        for (int i = 0; i < _classNames.size(); i++)
        {
            cout << "class name for class ID " << i << ": " << _classNames[i] << endl;
        }
    }

    //Parameterized Constructor
    Yolov5ObjectDetector(string weightsPath, string namesPath, float modelInputWidth, float modelInputHeight, float confidence, float nmsThreshold, bool useGpu)
    {
        _weightsPath = weightsPath;
        _namesPath = namesPath;
        _modelInputWidth = modelInputWidth;
        _modelInputHeight = modelInputHeight;
        _confidence = confidence;
        _nmsThreshold = nmsThreshold;
        _device = "cpu";
        if (useGpu) {
            _device = "gpu";
        }
        loadWeights();
    }

    void loadWeights() {

        // Load class names
        ifstream ifs(_namesPath);
        string line;

        while (getline(ifs, line))
        {
            _classNames.push_back(line);
        }

        // Load the ONNX model
        _net = readNet(_weightsPath);
        if (_device == "cpu")
        {
            cout << "Using CPU device" << endl;
            _net.setPreferableBackend(DNN_TARGET_CPU);
        }
        else if (_device == "gpu")
        {
            cout << "Using GPU device" << endl;
            _net.setPreferableBackend(DNN_BACKEND_CUDA);
            _net.setPreferableTarget(DNN_TARGET_CUDA);
        }
    }

    bool detect(Mat& inputImage, ModelDetections& results, bool displayResults) {
        // Check if the aspect ratio of the input image is almost the same as the aspect ratio of the model input size
        // if not, then the input image will be resized without keeping its aspect ratio when it 
        // is passed to the model, and this may lead to inaccurate detection
        float aspectRatioDiff = (inputImage.cols * _modelInputHeight) / (inputImage.rows * _modelInputWidth) - 1;
        if (abs(aspectRatioDiff) > 0.1) {
            string warning = format("The input image has a different aspect ratio: %.2f that the model! The results may not be accurat", inputImage.cols / (1.0 * inputImage.rows));
            cout << warning << endl;
            return false;
        }

        // Convert the input image to a blob, scale 
        Mat blob;
        // the last 3 arguments are the image mean = (0, 0, 0), swapRB = True, crop = False
        blobFromImage(inputImage, blob, 1. / 255., Size(_modelInputWidth, _modelInputHeight), Scalar(), true, false);

        _net.setInput(blob);

        // Forward propagate
        vector<Mat> detections;
        _net.forward(detections, _net.getUnconnectedOutLayersNames());
        results = _postProcess(inputImage, detections, displayResults);

        return true;
    }

    bool detectByCropping(Mat& inputImage, ModelDetections& results, const vector<Rect>& cropCorners, float nmsThresholdForRemovingDuplicates, bool displayResults)
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
            Rect corners;
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
        vector<vector<Rect>> boxes;
        vector<vector<int>> labels;
        vector<vector<float>> scores;


        // Combine the results, filter them based on the score, and update the coordinates of the bounding boxes for applying NMS later
        // A list to keep track of cropped sub-images with at least one object detection
        vector<int>  cropIdsWithDetection;
        for (int i = 0; i < cropCorners.size(); i++) {
            Rect corners;
            corners = cropCorners[i];
            //  Enlarge the crop if necessary to make all the same size
            Rect roi = Rect(corners.x, corners.y, cropWidth, cropHeight);
            // Crop the image and run the model
            Mat croppedImage = inputImage(roi);
            ModelDetections outputs;
            bool success = detect(croppedImage, outputs, false);

            if (!success || outputs.scores.size() == 0)
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

            for (int j = 0; j < outputs.boxes.size(); j++)
            {
                if (outputs.boxes[j].x < 4 || outputs.boxes[j].y < 4 || (outputs.boxes[j].x + outputs.boxes[j].width) >(cropWidth - 4) || (outputs.boxes[j].y + outputs.boxes[j].height) >(cropHeight - 4))
                {
                    outputs.scores[j] = _confidence;
                }

                // Now update/shift the boxes to the original image cooridnate
                outputs.boxes[j].x += corners.x;
                outputs.boxes[j].y += corners.y;

            }
            cropIdsWithDetection.push_back(i);
            boxes.push_back(outputs.boxes);
            labels.push_back(outputs.labels);
            scores.push_back(outputs.scores);
        }

        // No object detected, return
        if (cropIdsWithDetection.size() == 0)
        {
            return true;
        }

        // lists to contain the detections/results
        vector<Rect> outBoxes;
        vector<int> outLabels;
        vector<float> outScores;

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
            vector<Rect> cropBoxes = boxes[idx];

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
            vector<Rect> restBoxes;
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
                    vector<Rect> cropBoxesOfThisLabel;
                    vector<Rect> restBoxesOfThisLabel;
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

        if (displayResults)
        {
            // make a copy of the input image
            Mat debugImage = inputImage.clone();
            for (int i = 0; i < outBoxes.size(); i++)
            {
                Rect box = outBoxes[i];

                int left = box.x;
                int top = box.y;
                int width = box.width;
                int height = box.height;
                // Draw bounding box.
                rectangle(debugImage, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

                // Get the label for the class name and its confidence.
                string label = format("%.2f", outScores[i]);
                label = _classNames[outLabels[i]] + ":" + label;
                // Draw class labels.
                addDetection(debugImage, label, left, top);
            }
            results = ModelDetections(outBoxes, outLabels, outScores, debugImage);
        }
        else
        {
            results = ModelDetections(outBoxes, outLabels, outScores);
        }
        return true;
    }
};


int main(int argc, char** argv)
{

    cout << "USAGE : <path_to_image> " << endl;
    cout << "USAGE : <path_to_image> <device (cpu/gpu)>" << endl;
    cout << "USAGE : <path_to_image> <device (cpu/gpu)> <bit_depth>" << endl;
    cout << "USAGE : <path_to_image> <device (cpu/gpu)> <bit_depth> <normalize (true/false)>" << endl;

    string imageFile;
    bool useGpu = false;
    int bitDepth = BIT_DEPTH;
    bool normalize_image = false;


    // Take arguments from commmand line
    if (argc == 2)
    {
        imageFile = argv[1];
    }
    else if (argc == 3)
    {
        imageFile = argv[1];
        if ((string)argv[2] == "gpu")
            useGpu = true;
    }
    else if (argc == 4)
    {
        imageFile = argv[1];
        if ((string) argv[2] == "gpu")
            useGpu = true;
        bitDepth = stoi(argv[3]);

    }
    else if (argc == 5)
    {
        imageFile = argv[1];
        if ((string) argv[2] == "gpu")
            useGpu = true;
        bitDepth = stoi(argv[3]);
        if ((string) argv[4] == "true")
            normalize_image = true;
    }

    cout << "Image name (arg 1): " << imageFile << endl;
    cout << "Use of GPU (arg 2): " << useGpu << endl;
    cout << "Bit depth (arg 3) " << bitDepth << endl;
    cout << "Normalize flag (arg 4): " << normalize_image << endl;


    Yolov5ObjectDetector detector = Yolov5ObjectDetector(useGpu);
    
    // This part is not needed in the final code, it is included to 
    // load the model weight to GPU and run it on a test image for an accurate runtime measurement
    Mat testImage = Mat(Size(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), CV_8U, Scalar(0));
    cvtColor(testImage, testImage, COLOR_GRAY2BGR);
    ModelDetections testOut;
    bool testSuccess = detector.detect(testImage, testOut, false);

    // Load image
    Mat image;
    image = imread(imageFile, IMREAD_UNCHANGED);
  
    cout << "Input image Height: " << image.size().height << endl;
    cout << "Input image Width: " << image.size().width << endl;
    cout << "Input image Type: " << image.type() << endl;
    cout << "Input image Channels: " << image.channels() << endl;

    auto start = chrono::system_clock::now();

    if (image.channels() != 1)
    {
        // The image is a 3-channel image (This is not expected for ix-81 microscope and Breadboard images and should never happen)
        // Convert the image to gray scale for normalization/scaling (if needed)
        cvtColor(image, image, COLOR_BGR2GRAY);
    }

    // Change the matrix format to float before scaling
    image.convertTo(image, CV_64F);
    // The model expects 0-255 3-channel images, scale with bit-depth
    image = (255.0 / (exp2(bitDepth) - 1)) * image;

    if (normalize_image)
    {
        // Find the min and max values of the image
        // scale the matrix such that the min is mapped to 0 and the max is mapped to 255
        // we can use normalize function from OpenCV directly
        
        // double minVal;
        // double maxVal;
        // minMaxLoc(image, &minVal, &maxVal);
        // image = image - Mat(image.size(), image.type(), Scalar(minVal));
        // image = (255.0 / (maxValue - minValue)) * image;
        normalize(image, image, 0, 255, NORM_MINMAX);
    }
    // change back to unsigned int
    image.convertTo(image, CV_8U);

    // The model expects a 3 channel input image, expand the dimensions
    cvtColor(image, image, COLOR_GRAY2BGR);

    ModelDetections outputs;
    bool success = false;
    float scaleFactor = 1.0;
    if (image.rows == INPUT_IMAGE_HEIGHT && image.cols == INPUT_IMAGE_WIDTH)
    {
        success = detector.detect(image, outputs, true);
    }
    else
    {
        if (image.cols == IMAGE_RESOLUTION_IX81[0] && image.rows == IMAGE_RESOLUTION_IX81[1])
        { 
            // ix-81 microscope image
            Mat resizedImage;
            resize(image, resizedImage, Size(RESIZED_IMAGE_IX81[0], RESIZED_IMAGE_IX81[1]));
            scaleFactor = ((float)IMAGE_RESOLUTION_IX81[0]) / RESIZED_IMAGE_IX81[0];
            success = detector.detectByCropping(resizedImage, outputs, CROP_CORNERS_IX81, 0.1, true);
        }
        else if (image.cols == IMAGE_RESOLUTION_BB2[0] && image.rows == IMAGE_RESOLUTION_BB2[1])
        {
            // Breadboard image
            Mat resizedImage;
            resize(image, resizedImage, Size(RESIZED_IMAGE_BB2[0], RESIZED_IMAGE_BB2[1]));
            scaleFactor = ((float) IMAGE_RESOLUTION_BB2[0]) / RESIZED_IMAGE_BB2[0];
            success = detector.detectByCropping(resizedImage, outputs, CROP_CORNERS_BB2, 0.1, true);
        }
        else
        {
            cout << "Unsupported input image size!" << endl;
        }
    }

    if (success)
    {
        // Scale the bounding boxes back to the image resolution
        for (int i = 0; i < outputs.boxes.size(); i++) 
        {
            outputs.boxes[i].x *= scaleFactor;
            outputs.boxes[i].y *= scaleFactor;
            outputs.boxes[i].width *= scaleFactor;
            outputs.boxes[i].height *= scaleFactor;
        }

        auto end = chrono::system_clock::now();
        auto elapsed = end - start;
        string elapsedTime = format("Running YOLOv5 model took : %.2f ms", (elapsed.count() / 10000.0));
        cout << elapsedTime << endl;

        cout << "Successfully run the model!" << endl;
        cout << "Number of detected objects: " << outputs.boxes.size() << endl;
        // Save the debug image, the debug image is resized here
        if (outputs.boxes.size() > 0)
        {
            imwrite("results.jpg", outputs.debugImage);
        }
    }
    else
    {
        cout << "Failed to run the model!" << endl;
    }

    return 0;
}