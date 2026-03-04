import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from preprocess import X_train, X_test, y_train, y_test

# Get the correct number of classes
num_classes = y_train.shape[1]  # Ensure it dynamically matches your dataset

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True  
)
datagen.fit(X_train)

# Improved CNN Model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Ensure it matches dataset classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Learning Rate Scheduler
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# Train Model with Class Weights
model = build_model((128, 128, 1), num_classes)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,  
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[lr_reduction]
)

# Save the trained model
model.save("models/alzheimers_model.h5")
print("✅ Model saved successfully!")
