import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:\\Users\\Piyush\\Desktop\\diabetes\\dataset\\diabetes.csv")  # Ensure the CSV file is inside the dataset folder

# Check if the dataset is loaded correctly
print("Dataset Loaded Successfully! First 5 rows:")
print(df.head())

# Split data into features (X) and target (y)
X = df.drop(columns=["Outcome"])  # Features (all except the target column)
y = df["Outcome"]  # Target (diabetes positive = 1, negative = 0)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("models/diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Training Complete! ✅ Model saved as 'diabetes_model.pkl'")
