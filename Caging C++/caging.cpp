// Include Libraries
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "Yolov5ObjectDetector.h"

// Namespaces
using namespace std;

const vector<int> IMAGE_RESOLUTION_IX81 = { 2000, 1600 }; // Image resolution for ix-81 microscope
const vector<int> IMAGE_RESOLUTION_BB2 = { 4512, 4512 }; // Image resolution for Breadboard (BB2)
const vector<int> RESIZED_IMAGE_IX81 = { 1000, 800 }; // // The size of resized image for ix-81 images before cropping and running the model on the crops
const vector<int> RESIZED_IMAGE_BB2 = { 2880, 2880 }; // The size of resized image for Breadboard images before cropping and running the model on the crops
// Crop sizes for ix-81
const vector<cv::Rect> CROP_CORNERS_IX81 = { cv::Rect(0, 0, 640, 640 ), cv::Rect(0, 160, 640, 640), cv::Rect(360, 0, 640, 640), cv::Rect(360, 160, 640, 640) };
// Crop sizes for Breadboard (BB2)
const vector<cv::Rect> CROP_CORNERS_BB2 = {
    cv::Rect(0, 0, 640, 640),    cv::Rect(0, 560, 640, 640),    cv::Rect(0, 1120, 640, 640),    cv::Rect(0, 1680, 640, 640),    cv::Rect(0, 2240, 640, 640),
    cv::Rect(560, 0, 640, 640),  cv::Rect(560, 560, 640, 640),  cv::Rect(560, 1120, 640, 640),  cv::Rect(560, 1680, 640, 640),  cv::Rect(560, 2240, 640, 640),
    cv::Rect(1120, 0, 640, 640), cv::Rect(1120, 560, 640, 640), cv::Rect(1120, 1120, 640, 640), cv::Rect(1120, 1680, 640, 640), cv::Rect(1120, 2240, 640, 640),
    cv::Rect(1680, 0, 640, 640), cv::Rect(1680, 560, 640, 640), cv::Rect(1680, 1120, 640, 640), cv::Rect(1680, 1680, 640, 640), cv::Rect(1680, 2240, 640, 640),
    cv::Rect(2240, 0, 640, 640), cv::Rect(2240, 560, 640, 640), cv::Rect(2240, 1120, 640, 640), cv::Rect(2240, 1680, 640, 640), cv::Rect(2240, 2240, 640, 640) 
};
// Bit depth for both ix-81 and Breadboard images
const int BIT_DEPTH = 14;


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
    cv::Mat testImage = cv::Mat(cv::Size(detector.getInputHeight(), detector.getInputWidth()), CV_8U, cv::Scalar(0));
    cv::cvtColor(testImage, testImage, cv::COLOR_GRAY2BGR);
    // Outputs
    vector<cv::Rect> testOutBoxes;
    vector<int> testOutLabels;
    vector<float> testOutScores;
    bool testSuccess = detector.detect(testImage, testOutBoxes, testOutLabels, testOutScores);

    // Load image
    cv::Mat image;
    image = cv::imread(imageFile, cv::IMREAD_UNCHANGED);
  
    cout << "Input image Height: " << image.size().height << endl;
    cout << "Input image Width: " << image.size().width << endl;
    cout << "Input image Type: " << image.type() << endl;
    cout << "Input image Channels: " << image.channels() << endl;

    auto start = chrono::system_clock::now();

    if (image.channels() != 1)
    {
        // The image is a 3-channel image (This is not expected for ix-81 microscope and Breadboard images and should never happen)
        // Convert the image to gray scale for normalization/scaling (if needed)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
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
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
    }
    // change back to unsigned int
    image.convertTo(image, CV_8U);

    // The model expects a 3 channel input image, expand the dimensions
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    
    // Outputs
    vector<cv::Rect> outBoxes;
    vector<int> outLabels;
    vector<float> outScores;
    cv::Mat debugImage;
    bool success = false;
    float scaleFactor = 1.0;
    if (image.rows == detector.getInputHeight() && image.cols == detector.getInputWidth())
    {
        success = detector.detect(image, outBoxes, outLabels, outScores, debugImage);
    }
    else
    {
        if (image.cols == IMAGE_RESOLUTION_IX81[0] && image.rows == IMAGE_RESOLUTION_IX81[1])
        { 
            // ix-81 microscope image
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(RESIZED_IMAGE_IX81[0], RESIZED_IMAGE_IX81[1]));
            scaleFactor = ((float)IMAGE_RESOLUTION_IX81[0]) / RESIZED_IMAGE_IX81[0];
            success = detector.detectByCropping(resizedImage, CROP_CORNERS_IX81, 0.1, outBoxes, outLabels, outScores, debugImage);
        }
        else if (image.cols == IMAGE_RESOLUTION_BB2[0] && image.rows == IMAGE_RESOLUTION_BB2[1])
        {
            // Breadboard image
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(RESIZED_IMAGE_BB2[0], RESIZED_IMAGE_BB2[1]));
            scaleFactor = ((float) IMAGE_RESOLUTION_BB2[0]) / RESIZED_IMAGE_BB2[0];
            success = detector.detectByCropping(resizedImage, CROP_CORNERS_BB2, 0.1, outBoxes, outLabels, outScores, debugImage);
        }
        else
        {
            cout << "Unsupported input image size!" << endl;
        }
    }

    if (success)
    {
        // Scale the bounding boxes back to the image resolution
        for (int i = 0; i < outBoxes.size(); i++) 
        {
            outBoxes[i].x *= scaleFactor;
            outBoxes[i].y *= scaleFactor;
            outBoxes[i].width *= scaleFactor;
            outBoxes[i].height *= scaleFactor;
        }

        auto end = chrono::system_clock::now();
        auto elapsed = end - start;
        string elapsedTime = cv::format("Running YOLOv5 model took : %.2f ms", (elapsed.count() / 1000000.0));
        cout << elapsedTime << endl;

        cout << "Successfully run the model!" << endl;
        cout << "Number of detected objects: " << outBoxes.size() << endl;
        // Save the debug image, the debug image is resized here
        if (outBoxes.size() > 0)
        {
            cv::imwrite("results.jpg", debugImage);
        }
    }
    else
    {
        cout << "Failed to run the model!" << endl;
    }

    return 0;
}
