## Implementation of Neural Network, Support Vector Machine, and Decision Tree in C++

This is an assignment for the Advanced Artificial intelligence course. The required is to implement neural network (NN), support vector machine (SVM), and Decision tree (DT) algorithms in C++. 
For NN & DT we must implement from scratch. 

The goal is to understand the basics of these algorithms. Notice that this implementation is the basic to understand the idea of each algorithm. Maybe it’s not efficient enough, but it’s a good start and can be improved; especially most available codes for these algorithms are in Python and rare using C++.

### Requirements:
- Visual Studio 2017
-	OpenCV 

### The project code contains the following files:
-	 **main.cpp:**
The main function for running the program, it run the selected algorithm on the selected dataset.
-	**overwritesHeader.h & overwritesHeader.cpp :**
    - Useful functions to use for matrix mathematical operations (overwrites +,-..., Etc.).
    -	Used mostly in neural networks functions.
-	 **otheUsefulrFun.h & otheUsefulrFun.cpp:**
     - Contains common functions for the three models such as:
       - Read data files.
       - Normalize data.
       - Shuffle data.
-	**nn_header.h & nn_fun.cpp:**  Neural Network functions.
-	**svm_header.h & svm_fun.cpp:**  Support Vector Machine functions.
-	**dt_header.h & dt_fun.cpp:**   Decision Tree functions




### To run this code:
- Download files in code folder, and the datasets files from data folder.
-	Create a new empty project in Visual Studio 2017, and add code files to it.
-	Copy datasets files to the same directory of code files.
- Add Open CV following these steps:
  - download OpenCV library from the following link:
    https://sourceforge.net/projects/opencvlibrary/
  -	 Extract it to C drive.
  -	From advanced system settings, environment variables, Add C:\opencv\build\x64\vc15\bin to path variables.
  -	In your Visual Studio C++ project, project properties, do the following (after any step save by clicking apply):
    -	Change the mode to debug.
    -	In C/C++ configurations, in General tab, add C:\opencv\build\include to Additional include Directories.
    -	In Linker configurations, General tab, add C:\opencv\build\x64\vc15\lib to Additional Libraries Directories.
    -	In Linker configurations, Input tab, copy this file name opencv_world400d.lib into Additional dependencies.
    -	Change the mode to release.
    -	In Linker configurations, Input tab, copy this file name opencv_world400.lib into Additional dependencies.
    -	Click Ok to save all settings.
    - Change the mode from x86 to x64 in the tool menu before build and compile.
    -	Note: For reference, this video shows the above steps: https://www.youtube.com/watch?v=M-VHaLHC4XI

-	Just run the program, choose the dataset to work on, the algorithm you want to run, and adjust parameters as you want.
