import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "disease_symptom.csv")
ML_DIR = os.path.join(BASE_DIR, "ml")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Clean column names and values
df.columns = df.columns.str.strip()
df = df.fillna("")

# Get all unique symptoms
symptom_columns = [col for col in df.columns if col != "Disease"]
all_symptoms = set()
for col in symptom_columns:
    df[col] = df[col].str.strip()
    all_symptoms.update(df[col].unique())

all_symptoms.discard("")
all_symptoms = sorted(list(all_symptoms))
print(f"Total unique symptoms found: {len(all_symptoms)}")

# Convert each row into a binary vector
# 1 if symptom is present, 0 if not
def row_to_vector(row):
    present = set(row[symptom_columns].values)
    return [1 if s in present else 0 for s in all_symptoms]

print("Building feature matrix...")
X = df.apply(row_to_vector, axis=1, result_type="expand")
X.columns = all_symptoms

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"].str.strip())

print(f"Diseases found: {list(le.classes_)}")
print(f"Total diseases: {len(le.classes_)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model, encoder, and symptom list
joblib.dump(model, os.path.join(ML_DIR, "symptom_model.pkl"))
joblib.dump(le, os.path.join(ML_DIR, "label_encoder.pkl"))
joblib.dump(all_symptoms, os.path.join(ML_DIR, "symptom_list.pkl"))

print("Model saved successfully!")
print(f"Files saved in: {ML_DIR}")