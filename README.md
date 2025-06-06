# 🤟 Sign Language Detector

A real-time hand gesture recognition system that converts sign language into readable text or speech using computer vision and machine learning.

---

## 🧠 Features

- 📷 Collects custom hand gesture images using webcam
- 🧠 Trains a custom ML model using Random Forest
- 💡 Real-time detection and classification of hand signs
- 🔊 Optional text-to-speech feedback for recognized signs

---

## 💻 Tech Stack

- **Languages & Libraries**: Python, OpenCV, NumPy, scikit-learn, Tkinter
- **ML Model**: Random Forest Classifier
- **Others**: pickle, os, cv2, time

---

## 📁 Project Structure

sign-language-detector/
├── collect_imgs.py # Capture and store gesture images  
├── create_dataset.py # Preprocess images and create labeled data  
├── train_classifier.py # Train ML model and save as model.p  
├── inference_classifier.py # Use webcam to detect gestures live  
├── requirements.txt # Python dependencies  
├── model.p # Trained classifier (small enough to include)  
├── data.pickle # Labeled dataset (optional for upload)  
├── README.md # This file  


---

## ⚠️ Notes Before You Start

- **Python 3.10 is required**. This project will NOT work on Python 3.11 or 3.13 due to `mediapipe` compatibility issues.
- You need a **working webcam**.
- The project assumes a trained `model.p` file already exists. If you're cloning this repo fresh, it will run without re-training.

---

## 🔧 Setup Instructions

### 1️⃣ Install Python 3.10

👉 Download it here: [Python 3.10](https://www.python.org/downloads/release/python-3100/)  
✅ Check "Add to PATH" during installation

### 2️⃣ Clone the Repo
git clone https://github.com/your-username/sign-language-detector-python.git
cd sign-language-detector-python
python collect_imgs.py

3️⃣ Create Virtual Environment
py -3.10 -m venv venv
.\venv\Scripts\activate

4️⃣ Install Dependencies
pip install opencv-python mediapipe==0.10.9 scikit-learn

You can also run:
pip install -r requirements.txt

5️⃣ Run the Project
python inference_classifier.py

The webcam will open.
Show a trained hand gesture (A, B, C, etc.)
Prediction will appear in the console.
For privacy and size reasons, the data/ folder (with training images) is not included.

If you'd like to retrain the model:

python collect_imgs.py   # Collect new images
python train_classifier.py  # Train model.p again
Make sure you follow the same class labeling and structure used originally.

---

## 👩‍💻 Author

**Rasagna Rayasam**  
Aspiring AI/ML Developer | Passionate about building tech that helps people  
📫 [Connect on GitHub](https://github.com/RasagnaRayasam)

---

Thanks for checking out this project! ⭐ If you liked it, drop a star on the repo to support my work.










