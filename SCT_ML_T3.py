import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset"
IMG_SIZE = 64

X = []
y = []

def load_images(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog(gray,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys')

        X.append(features)
        y.append(label)

# Cats = 0
load_images(os.path.join(DATASET_PATH, "cats"), 0)

# Dogs = 1
load_images(os.path.join(DATASET_PATH, "dogs"), 1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Total images loaded:", len(X))
print("Total labels:", len(y))
print("Program finished")
