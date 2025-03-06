# 🖐️ Sign Language Detection  

This project is a **real-time sign language detection system** that detects and classifies hand signs using **OpenCV, cvzone, and a deep learning model (Keras)**. The system captures hand gestures from a webcam, processes them, and predicts the corresponding sign using a pre-trained model.  

## 📌 Features  
✅ **Real-time Hand Detection** – Uses OpenCV and cvzone to track hand movements.  
✅ **Sign Classification** – Recognizes hand signs (`A, B, C, Y, L, 1, 5`).  
✅ **Deep Learning Model Integration** – Uses a Keras model (`keras_model.h5`) for classification.  
✅ **Image Preprocessing** – Crops, resizes, and normalizes hand images for better model accuracy.  
✅ **User-Friendly Visualization** – Displays bounding boxes and predictions on the webcam feed.  

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Libraries Used**: OpenCV, cvzone, NumPy, TensorFlow/Keras  
- **Deep Learning Model**: Trained on a custom dataset for sign language recognition  
- **Hardware Requirement**: Webcam for real-time gesture detection  

## 🚀 How It Works  
1. **Captures video from a webcam** and detects a single hand using `HandDetector` from cvzone.  
2. **Extracts the hand region**, resizes it to `300x300`, and processes it to match the model’s input size.  
3. **Passes the image to a trained deep learning model** (`keras_model.h5`), which predicts the corresponding sign.  
4. **Displays the detected sign** on the webcam feed with bounding boxes and text annotations.  
5. **Terminates when 'q' is pressed** to stop the webcam feed.  

