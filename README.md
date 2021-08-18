# Detecting Driver Drowsiness using Behavioral measures and Representation learning: A Deep Learning-based Technique

![](http://www.clipartsuggest.com/images/730/know-drowsy-driving-has-been-a-topic-before-i-wanted-to-reiterate-g97Upj-clipart.gif)


## Table of contents

1.  [Project Description](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#project-description)
1.  [Configuration Instructions](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#configuration-instructions)
    1.  [System Requirements](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#system-requirements)
    1.  [Tools and Library Requirements](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#tools-and-library-requirements)
1.  [Installation Instructions](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#installation-instructions)
1.  [Operating Instructions](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#operating-instructions)
1.  [Manifesting Directory structure](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#manifesting-directory-structure)
1.  [Copyrights Information](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#copyrights-information)
1.  [Contact List](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#contact-list)
1.  [Bugs](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#bugs)
1.  [Troubleshooting](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#troubleshooting)
1.  [Credits](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#credits)
1.  [Changes logs/news](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md#changes-logsnews)





## Project Description
The inclining in road accidents is one of the most prevailing problems around the world. The growing number of vehicles on the road has paved a way for many inefficient and inattentive drivers. The key to safe driving and prevention of accidents is predominantly dependent on the driver. Preventing accidents by drowsiness need a technique to detect the factors that cause sleepiness in driver. Drowsiness or sleepiness is a biological state where the body is in transition from an awake state to a sleeping state in which the level of consciousness is reduced due to lack of sleep or fatigue, this causes the driver to fall into sleep quietly. During this stage, a driver is unable to take control of the vehicle and loses consciousness. 

According to statistics from the US NHTSA out of 1,00,000 accidents reported annually, 76,000 crashes are due to drowsiness. In Australia, about 6% of accidents happen due to drowsiness and UK reports approximately 20% of the accidents by fatigue. These statistics infer the importance of a drowsiness detection system is unavoidable and motivated to build a detection system as a precautionary measure to avoid road accidents. A common solution to the drowsiness detection problem is to develop a model that detects the sleepiness in drivers in an early stage to prevent accidents by considering several common signs and alert the driver if any of the signs are frequently detected. This project proposes to develop a robust model to detect the sleepiness in drivers by capturing behavior-based images such as yawning, eye blinking, and head-nodding states. 

## Configuration Instructions
The [Project](https://github.com/sowmi06/Driver_Drowsiness_Detection.git) requires the following tools and libraries to run the [source code](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Source_Code/Drowsiness_Detection.py)
### System Requirements

- GPU based CUDA®-enabled card (Ubuntu and Windows)
 
- [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/install/)
    - Python version 3.6.0 – 3.9.0
    - pip 19.0 or later 
    - Ubuntu 16.04 or later (64-bit)
    - macOS 10.12.6 (Sierra) or later (64-bit)
    - Windows 7 or later (64-bit) 
 
- Python IDE (to run ".py" file)
    - [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), [Spyder](https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/spyder/doc/installation.html) or [VS code](https://code.visualstudio.com/download)

### Tools and Library Requirements 
- [TensorFlow](https://www.tensorflow.org/install/pip)
   
   Installs tensorflow in virtual environment using pip:
        
        pip install --upgrade tensorflow
        
   To check your tensorflow installation:
   
        python -m pip show tensorflow # to see which version and where tensorflow is installed
        
    
- [Numpy](https://numpy.org/install/)

  Numpy releases are available as wheel packages for macOS, Windows and Linux on [PyPI](https://pypi.org/project/numpy/). 
  
  Install numpy using pip:
        
        pip install numpy
                
  Install numpy using Conda packages:

        conda install numpy
  
  To check your numpy installation:
   
        python -m pip show numpy # to see which version and where numpy is installed

   

- [Scikit-learn](https://scikit-learn.org/stable/install.html) 
  
  Install scikit-learn using pip:
        
        pip install -U scikit-learn
                
  To check your scikit-learn installation:

        python -m pip show scikit-learn  # to see which version and where scikit-learn is installed
     

- [OpenCV](https://pypi.org/project/opencv-python/)

  OpenCv releases are available as wheel packages for macOS, Windows and Linux on [PyPI](https://pypi.org/project/opencv-python/).
  
  Install OpenCv using pip:
        
        pip install opencv
                
  Install OpenCV using Conda packages:

        conda install opencv
  
  To check your installation:

        python -m pip show opencv  # to see which version and where opencv is installed


- [Matplotlib](https://matplotlib.org/stable/users/installing.html)

  Matplotlib releases are available as wheel packages for macOS, Windows and Linux on [PyPI](https://pypi.org/project/matplotlib/).
  
  Install it using pip:
        
        python -m pip install -U pip
        python -m pip install -U matplotlib
        
  Install it using Conda packages:

        conda install matplotlib
   

## Installation Instructions
To work with the project code
- Clone the [Driver_Drowsiness_Detection](https://github.com/sowmi06/Driver_Drowsiness_Detection.git) repository into your local machine from this link : https://github.com/sowmi06/Driver_Drowsiness_Detection.git

- Follow the same directory structure from the cloned repository. 


## Operating Instructions

The following are the steps to replicate the exact results acquired from the project:

- Satisify all the system and the tool, libraries requirements.
- Clone the [Driver_Drowsiness_Detection](https://github.com/sowmi06/Driver_Drowsiness_Detection.git) repository into your local machine. 
- Run the [Drowsiness_Detection.py](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Source_Code/Drowsiness_Detection.py) in a python IDE to get the final output
- Follow the same directory structure from the cloned repository.


## Manifesting Directory structure

The following directory structure is required to replicate exact results acquired from the project:

### Directory layout to repicate results

    .
    ├── Cascade_Classifier                 
    │   ├── haarcascade_eye.xml
    |   └── haarcascade_frontalface_default.xml
    |
    ├── Dataset                 
    │   ├── Eyes
    |   |    ├── Close_eye                    
    |   |    |    └── ...  # raw images 
    |   |    └── Open_eye
    |   |         └── ...  # raw images  
    │   └── Face
    |        ├── no_yawn
    |        |    └── ...  # raw images 
    |        └── yawn
    |             └── ...  # raw images 
    |  
    ├── Source_Code   
    │   ├── Drowsiness_Detection.py 
    │   └── face_eye_detector.py
    |
    ├── LICENSE                     
    └── README.md




### Directories and Files

[Cascade_Classifier](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Cascade_Classifier) - A folder containing Haar-Cascade Classifier ".xml" files from official [OpenCV repository](https://github.com/opencv).

[haarcascade_eye.xml](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Cascade_Classifier/haarcascade_eye.xml) - A ".xml" file to detect the eyes from image.

[haarcascade_frontalface_default.xml](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Cascade_Classifier/haarcascade_frontalface_default.xml)- A ".xml" file to detect the faces from image.

[Dataset](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset) - A folder with two sub folders containing Eyes and Faces raw images.

[Eyes](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Eyes) - A sub-folder containing raw images of [open](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Eyes/Open_eye) and [close eye](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Eyes/Close_eye).

[Face](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Face) - A sub-folder containing raw images of [yawn](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Eyes/yawn) and [no yawn](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Dataset/Eyes/no_yawn).

[Source_Code](https://github.com/sowmi06/Driver_Drowsiness_Detection/tree/main/Source_Code) - A folder containing source code to execude the output.

[Drowsiness_Detection.py](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Source_Code/Drowsiness_Detection.py) - A ".py" file containing the proposed model implementation of the drowsiness detection system.

[face_eye_detector.py](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/Source_Code/face_eye_detector.py) - A ".py" file containing face detector and preprocessing steps.

[Readme.md](https://github.com/sowmi06/Driver_Drowsiness_Detection/blob/main/README.md) - Readme file to execute the project. 



## Copyrights Information

## Contact List
## Bugs
## Troubleshooting
## Credits
## Changes logs/news

## Reference
- Tensorflow Installation : https://www.tensorflow.org/install/pip#virtual-environment-install
- Numpy Installation : https://numpy.org/install/
- Scikit learn Installation : https://scikit-learn.org/stable/install.html
- OpenCv Installation : https://pypi.org/project/opencv-python/
- Mathplotlib Installation: https://pypi.org/project/matplotlib/
- Haar cascade classifier souce files: https://github.com/opencv
