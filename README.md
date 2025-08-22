## Step 01 : Create Virtual Environment
__
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
###### This step focuses on collecting training data for each gesture (like yes, no, hello, etc.). Images of your hand are captured using a webcam and stored in separate folders for each class. 
``` python datacollection.py ```
###### to upload data to each Sign Gestures 


## Step 4: Train the Model
___
###### Once you’ve collected your dataset in the data/ folder, it’s time to train a machine learning model to recognize each gesture.

