import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Crop.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = data[['temperature', 'humidity', 'ph', 'rainfall', 'N', 'P', 'K']]

y = data['Cid']

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Create Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Create Crop Mapping
# -----------------------------
crop_mapping = dict(zip(data['Cid'], data['label']))

print("\nCrop Mapping:\n")

print(crop_mapping)

# -----------------------------
# Store All Artifacts
# -----------------------------
artifacts = {
    "model": model,
    "crop_mapping": crop_mapping,
    "accuracy": round(accuracy * 100, 2)
}

# -----------------------------
# Save All Artifacts
# -----------------------------
with open("artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nAll artifacts saved successfully as artifacts.pkl")