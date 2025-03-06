import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/5"
os.makedirs(folder, exist_ok=True)

labels = ["A", "B", "C", "Y", "L", "1", "5"]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture from camera.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        x_start, y_start = max(x - offset, 0), max(y - offset, 0)
        x_end, y_end = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
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

            prediction, index = classifier.getPrediction(imgWhite)
            print(f"Prediction: {labels[index]} ({prediction})")

            cv2.putText(img, labels[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            print("Error: Cropped image is empty.")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

