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
2. Press O
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
