# Pytorch Deploy Script and C++ Executable Program Construction


1. Train your network with Pytorch, and get a **.pth** model.

2. Get the torch trace model using **Production/TrachTorch.py**.

3. Config the **LibTorch** and **OpenCV** libs in your platform.

4. Write your Inference **.cpp** file. A example is given in **Production/LibTorch.cpp**.

5. Config the CMakeLists.txt. A example is given in **Production/CMakeLists.txt**.

6. Using Cmake commands in your command, like ```mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug/Release .. && make```

7. You can find and run  your executable file in build folder.


# Pytorch Deploy Script and Python Executable Program Construction

1. Train your network with Pytorch, and get a **.pth** model.

2. Get the onnx model from **.pth** using a example like **Production/Extra_Code_wo_Verify/ExportONNX.py**.

3. Try to run your **inference.py** to inference your model using a example like **Production/Extra_Code_wo_Verify/ONNX_Run_Python.py**.

4. Use command ```pyinstaller -F inference.py``` to get a executable file.


**!!!** Files in **Production/Extra_Code_wo_Verify** have not been verified in use.

**!!!** You can use VSCode/Terminal to try our scripts.



If you have any questions, please contact with me.
