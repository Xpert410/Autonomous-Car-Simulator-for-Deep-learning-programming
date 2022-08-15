# Autonomous-Car-Simulator-for-Deep-learning-programming

/requirements.txt = List of packages that needs to be installed to run the program

/Client = Contains all the resources needed for the hand gesture recognition and for communicating with the JetBot

/Client/hand_history_classifier = Contains resources such as the pre-trained model used for hand gesture recognition

/Client/hand_history_classifier/hand_classifier.hdf5 = The pre-trained model

/Client/hand_history_classifier/hand_classifier.tflite = The pre-trained model converted to TFLite

/Client/hand_history_classifier/hand_history.csv = Dataset of coordinates of the hand gestures collected that will be used to train the model

/Client/hand_history_classifier/hand_history_classifier.py = Used to allow the Jupyter Notebook to 

/Client/hand_history_classifier/hand_history_classifier.csv = Dataset labels used to train the model
0 = Left
1 = Right
2 = Throttle
3 = Brake
4 = Toggle Reverse
5 = Full Stop, not used on the JetBot

/Client/utils = Contains the cvfpscalc.py that will be used to display the FPS on the camera feed

/Server = Contains all resources needed to run the collision avoidance and for communicating with the client to control the JetBot

/Server/data_collection-Modified.ipynb = Contains the codes used to create the dataset that will be used to train the collision avoidance model

/Server/train_model_resnet18.ipynb = Contains the codes used to re-train the resnet18 model for collision avoidance 

/Server/Hand Gesture Recog.ipynb = Contains the codes for communicating with the client to control the JetBot. DOES NOT HAVE COLLISION AVOIDANCE

/Server/Hand Gest + Colli Avoid.ipynb = Contains the codes for communicating with the client to control the JetBot, and also have collision avoidance








