import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
offset = 20
imgSize = 300
counter = 0

folder = "D:/Document/SignRecognitionSystem/data/one"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

        # imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        # imgCropShape = imgCrop.shape

        # aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f'Saved Image :{counter}')
    elif key == ord("q"):
        break