## YOLOv5 Caging Model

**This repository contains the YOLOv5 Caging code converted to C++ and run using OpenCV DNN**

## Models

The same ONNX .onnx weight and .names files used for the Python code are used by this code.

## Pre-requisits

The code required OpenCV libraries built with CUDA support.  

## Run Code:

### CMAKE C++ Windows
```
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
copy Release\caging.exe ..
cd ..
caging <input_file> <gpu/cpu> <bit_depth> <normalize>
```

