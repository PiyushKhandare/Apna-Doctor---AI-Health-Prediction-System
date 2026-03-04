import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset function
def load_data(data_dir, img_size=(128, 128)):
    classes = os.listdir(data_dir)
    X, y = [], []
    class_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            image = cv2.resize(image, img_size)  # Resize
            X.append(image)
            y.append(class_map[cls])

    X = np.array(X) / 255.0  # Normalize
    X = X.reshape(-1, img_size[0], img_size[1], 1)
    y = to_categorical(y, num_classes=len(classes))  # Convert labels to one-hot encoding
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define dataset path
data_dir = "C:\\Users\\Piyush\\Desktop\\alzheimer\\Alzheimer dataset"
X_train, X_test, y_train, y_test = load_data(data_dir)
