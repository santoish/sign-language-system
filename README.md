# Sign Recognintion System 

##### This project allows you to **collect hand gesture data**, **train a deep learning model**, and **run real-time predictions** using your webcam. 

## Step 01 : Create Virtual Environment
___
```
python -m venv myenv
myenv\Scripts\activate
```

## Step 02 : Packages need to Install
___
```
opencv-python
cvzone
mediapipe
numpy
pyttsx3
tensorflow
```

## Step 3: Data Collection
___
 This step focuses on collecting training data for each gesture (like yes, no, hello, etc.). Images of your hand are captured using a webcam and stored in separate folders for each class. 
 Run this to upload data to each Sign Gestures

 
``` python datacollection.py ```


## Step 4: Train the Model
___
 Once you’ve collected your dataset in the data/ folder, it’s time to train a machine learning model to recognize each gesture. To Train the Model
 Run this
 After Training model it'll create a sign_model.h5 file

 
``` python train_model.py```


## Step 5: Run Live Prediction
___
 Once the model (sign_model.h5) is trained, you can test it in real time using your webcam.
 To run

 
``` python live_prediction.py ```


 Q to Quit
