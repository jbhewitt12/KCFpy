# KCF tracker in Python

This is a fork of the [Python implmentation of KCF](https://github.com/uoip/KCFpy). I have modified the code to run sequences from the [VOT Challenge](http://www.votchallenge.net/) and provide Accuracy, Robustness and Frames Per Second evaluation metrics.

The original paper:
> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista<br>
> TPAMI 2015

### Requirements
- Python 2.7
- NumPy
- Numba 
- OpenCV (ensure that you can `import cv2` in python)

I recommend installing Anaconda(for Python 2.7), and OpenCV 3.1(from [opencv.org](http://opencv.org/)).

### To Run:
- Install the required packages

```shell
git clone https://github.com/uoip/KCFpy.git
cd KCFpy
```

- Download a dataset from the [VOT Challenge.](http://www.votchallenge.net/) webpage. [eg. VOT2016 dataset](http://data.votchallenge.net/vot2016/vot2016.zip)
- copy the dataset into the 'VOTdataset' folder in KCFpy
- from the command window, call run.py with 1 paramater: The path to the sequence you wish to run from the 'VOTdataset' folder  
eg:

```shell
python run.py VOT2016/matrix
```

The figures will automatically be saved into the 'saved' folder. Comment out line 192 in run.py to stop this. 

### My setup
I used Anaconda for Python 3 and setup a python 2.7 virtual environment called 'pykcf' on Windows 10.

How I did this: 
```shell
conda create --name pykcf python=2.7.13 numpy numba scipy
activate pykcf
pip install openCV-python
```



