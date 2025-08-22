import cv2
import pyttsx3
import numpy as np
import time
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector


# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Labels (adjust according to your training classes)
labels = ["hello", "thanks", "yes", "no"]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300
offset = 20



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Preprocess and predict
        # imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = imgInput / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)
        prediction = model.predict(imgInput)
        classID = np.argmax(prediction)
        confidence = prediction[0][classID]

        """ prediction = model.predict(imgInput)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        current_time = time.time()

        if predicted_label != last_prediction or (current_time - last_spoken_time) > speak_delay:
            print("Predicted:", predicted_label)
            engine.say(predicted_label)
            engine.runAndWait()
            last_prediction = predicted_label
            last_spoken_time = current_time """
        

        # Display result
        cv2.putText(img, f'{labels[classID]} ({confidence*100:.2f}%)',
                    (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
