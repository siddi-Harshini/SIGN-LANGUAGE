import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import tensorflow as tf
import keras



import numpy as np
import os

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

detector = HandDetector(maxHands=1)

#classifier = Classifier("Model/keras_model_fixed.h5", "Model/labels.txt")


classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20
imgSize = 300

folder = "Data/5"
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0

labels = ["A", "B", "C" ,"Y","L" ,"1" ,"5"]


while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture from camera.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping boundaries are within the image size
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, img.shape[1])
        y_end = min(y + h + offset, img.shape[0])

        imgCrop = img[y_start:y_end, x_start:x_end]

        # Check if imgCrop is valid
        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:  # If height > width
                k = imgSize / h
                wCal = round(k * w)  # Use round instead of math.ceil
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize

                #prediction, index = classifier.getPrediction(imgWhite)

                prediction, index = classifier.getPrediction(img)
                print(prediction, index)


            else:  # If width >= height
                k = imgSize / w
                hCal = round(k * h)  # Use round instead of math.ceil
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

            # Display images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            print("Error: Cropped image is empty.")

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
