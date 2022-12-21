# YOLOv5 for Cell Detection

## YOLOv5 Inference
The provided Jupyter notebook shows how to run the trained YOLOv5 model on 2000x1600 13-bit color-depth microscope images for cell detection. Note that the model is trained on CellPose cell detections and not properly annotated images. So the detections at best can be as accurate as CellPose detections. Once the images are annotated properly, YOLOv5 should be trained again for better accuracy. 

The model requires two files:
- The ONNX weights file (.onnx file)
- The list of labels (.names file)

YOLOv5 is impleneted and trained in PyTorch framework. To run inference efficiently and without the need for PyTorch, the model is then converted to ONNX and is run using OpenCV dnn module below. 

To be able to run the converted model on GPU, OpenCV built with CUDA should be installed. Otherwise, the model runs on CPU with longer runtimes. The ONNX conversion and running the model with OpneCV 4.5.4 build with CUDA dnn has been tested with yolov5 release v6.2 and PyTorch 1.11.0 (ONNX opset 12 is used for conversion). OpenCV dnn may not be able to read and run ONNX models converted with newer versions of yolov5 (e.g., v7.0) or PyTorch (e.g., 1.12.0) for conversion. ONNX version is not important.    
