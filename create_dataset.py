import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Check if it's a directory
        for img_path in os.listdir(dir_path):
            img_path_full = os.path.join(dir_path, img_path)
            if os.path.isfile(img_path_full):  # Ensure it's a file, not another directory
                data_aux = []

                x_ = []
                y_ = []

                img = cv2.imread(img_path_full)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    # If only one hand is detected, pad the data to 84 features (42 for each hand)
                    if len(results.multi_hand_landmarks) == 1:
                        while len(data_aux) < 84:  # Pad to 84 features (42 for one hand)
                            data_aux.append(0)

                    # If two hands are detected, ensure 84 features (42 for each hand)
                    if len(results.multi_hand_landmarks) == 2:
                        while len(data_aux) < 84:  # Ensure 84 features (42 for each hand)
                            data_aux.append(0)

                    # Append the features and label
                    data.append(data_aux)
                    labels.append(dir_)

# Save the processed data and labels to pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
