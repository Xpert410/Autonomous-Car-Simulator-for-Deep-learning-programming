# Autonomous-Car-Simulator-for-Deep-learning-programming

/Hand Gesture Recognition = Contains all resources for the hand gesture recognition program

/Hand Gesture Recognition/app.py = Run hand gesture recognition program

/Hand Gesture Recognition/model/hand_history_classifier = Contains resources such as the pre-trained model used for hand gesture recognition

/Hand Gesture Recognition/model/hand_classifier.hdf5 = The pre-trained model

/Hand Gesture Recognition/model/hand_classifier.tflite = The pre-trained model converted to TFLite

/Hand Gesture Recognition/model/hand_history.csv = Dataset of coordinates of the hand gestures collected that will be used to train the model

/Hand Gesture Recognition/model/hand_history_classifier.py = Contains functions needed for the app.py to run

/Hand Gesture Recognition/model/hand_history_log.csv = Dataset labels used to train the model 
0 = Left 
1 = Right 
2 = Throttle 
3 = Brake
4 = Toggle Reverse 
5 = Full Stop, not used on the JetBot

/Hand Gesture Recognition/utils = Contains the cvfpscalc.py that will be used to display the FPS on the camera feed

# app.py controls

K = Log hand gesture coordinates and save to dataset

N = Normal mode

ESC = Close program

# Create dataset

1. Run app.py
2. Press K
3. Put your hand in the starting position of the hand gesture that you wish to collect
4. Do the hand gesture that you wish to collect
5. Press numbers 1 - 5 depending on what gesture you are collecting
  0 = Left
  1 = Right
  2 = Throttle
  3 = Brake
  4 = Toggle Reverse
  5 = Full Stop
Can add more by adding more labels in /Client/model/hand_history_classifier/hand_history_label.csv

# Carla Simulator + Hand Gesture

## Requirements

Software:
* Python 3.7
* Carla 0.9.12

Hardware (Minimum):
* CPU: Quad or Octa core
* GPU: Dedicated, as much memory as possible
* RAM: 16GB

## Installation
1. Download CARLA 0.9.12 from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.12/).

## Quick Start
* Launching the simulator: \CARLA_0.9.12\WindowsNoEditor\Carla.exe
* Launching the car: \CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\manual_control
* Adding dynamic weather: \CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\dynamic_weather
* Adding traffic and pedestrians: \CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\generate_traffic

# Object Detection
Link to the object detection colab is [here](https://colab.research.google.com/drive/1fMPs0Y7mw1gys9s6zMMe2rvcjV23bsoH?usp=sharing).

## How to use
1. Run the codes one by one and make sure there is a tick on the code before running the next
2. !train.py is for training the dataset
3. !detect is for object detection on a video or picture

# Video demo
https://www.youtube.com/watch?v=caQdbPbA_Oo

CARLA Simulator + Hand Gesture: https://youtu.be/9HP2e8fJD2k

CARLA Simulator Quick Start: https://youtu.be/VVKuHdc_W8U

