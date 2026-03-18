import joblib
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, "ml")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load trained model files once when server starts
print("Loading ML models...")
model = joblib.load(os.path.join(ML_DIR, "symptom_model.pkl"))
label_encoder = joblib.load(os.path.join(ML_DIR, "label_encoder.pkl"))
symptom_list = joblib.load(os.path.join(ML_DIR, "symptom_list.pkl"))

# Load disease description and precaution datasets
print("Loading disease datasets...")
df_description = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
df_precaution = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
df_severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

# Clean column names
df_description.columns = df_description.columns.str.strip()
df_precaution.columns = df_precaution.columns.str.strip()
df_severity.columns = df_severity.columns.str.strip()

# Build description lookup dictionary
# { "Fungal infection": "Fungal infection is a..." }
disease_descriptions = {}
for _, row in df_description.iterrows():
    disease = str(row["Disease"]).strip()
    description = str(row["Description"]).strip()
    disease_descriptions[disease.lower()] = description

# Build precaution lookup dictionary
# { "Fungal infection": ["keep area dry", "use antifungal cream", ...] }
disease_precautions = {}
for _, row in df_precaution.iterrows():
    disease = str(row["Disease"]).strip()
    precautions = []
    for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            precautions.append(val)
    disease_precautions[disease.lower()] = precautions

# Build symptom severity lookup
# { "itching": 1, "chest_pain": 7 }
symptom_severity = {}
for _, row in df_severity.iterrows():
    symptom = str(row["Symptom"]).strip().lower()
    try:
        weight = int(row["weight"])
    except Exception:
        weight = 1
    symptom_severity[symptom] = weight

print("All datasets loaded successfully!")


def get_all_symptoms() -> list:
    """Return full list of symptoms the model knows"""
    return symptom_list


def get_disease_description(disease: str) -> str:
    """Get plain English description of a disease"""
    return disease_descriptions.get(
        disease.lower(),
        f"{disease} is a medical condition. Please consult a doctor for more information."
    )


def get_disease_precautions(disease: str) -> list:
    """Get list of precautions for a disease"""
    return disease_precautions.get(
        disease.lower(),
        ["Rest well", "Drink plenty of water", "Consult a doctor", "Avoid self medication"]
    )


def calculate_severity_score(symptoms_input: list) -> int:
    """
    Calculate total severity score from symptoms
    based on Symptom-severity.csv weights
    """
    total = 0
    for symptom in symptoms_input:
        clean = symptom.strip().lower().replace(" ", "_")
        total += symptom_severity.get(clean, 1)
    return total


def calculate_risk(disease: str, confidence: float, severity_score: int) -> str:
    """
    Determine risk level based on:
    - Disease type
    - ML model confidence
    - Total symptom severity score
    """
    high_risk_diseases = [
        "heart attack", "paralysis (brain hemorrhage)", "aids",
        "tuberculosis", "hepatitis b", "hepatitis c", "hepatitis d",
        "hepatitis e", "hepatitis a", "dengue", "malaria", "pneumonia",
        "hypoglycemia", "alcoholic hepatitis"
    ]

    medium_risk_diseases = [
        "diabetes", "hypertension", "hyperthyroidism", "hypothyroidism",
        "chronic cholestasis", "jaundice", "typhoid", "bronchial asthma",
        "cervical spondylosis", "gastroenteritis", "peptic ulcer diseae",
        "urinary tract infection", "chicken pox",
        "dimorphic hemmorhoids(piles)"
    ]

    disease_lower = disease.lower()

    if any(d in disease_lower for d in high_risk_diseases):
        return "high"
    elif any(d in disease_lower for d in medium_risk_diseases):
        if severity_score >= 10:
            return "high"
        return "medium"
    else:
        if severity_score >= 15:
            return "medium"
        return "low"


def build_advice(disease: str, precautions: list, risk_score: str) -> str:
    """
    Build a simple, clear advice paragraph from disease info
    No external API needed — pure Python
    """
    description = get_disease_description(disease)

    if risk_score == "high":
        urgency = "This is a serious condition. Please see a doctor immediately without delay."
    elif risk_score == "medium":
        urgency = "Please consult a doctor soon. Do not ignore these symptoms."
    else:
        urgency = "Monitor your symptoms. See a doctor if symptoms get worse."

    precaution_text = " Also: " + ", ".join(precautions) + "." if precautions else ""

    advice = f"{description} {urgency}{precaution_text}"
    return advice


def predict_disease(symptoms_input: list) -> dict:
    """
    Main prediction function.
    Takes a list of symptom strings.
    Returns predicted disease + confidence + risk + advice + precautions.
    """

    # Clean input symptoms
    cleaned_input = [s.strip().lower().replace(" ", "_") for s in symptoms_input]

    # Build binary vector
    input_vector = []
    matched_symptoms = []

    for symptom in symptom_list:
        symptom_clean = symptom.strip().lower()
        if symptom_clean in cleaned_input:
            input_vector.append(1)
            matched_symptoms.append(symptom)
        else:
            input_vector.append(0)

    # Predict disease
    input_array = np.array(input_vector).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    # Get top 3 possible diseases
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = []
    for idx in top_indices:
        prob = probabilities[idx]
        if prob > 0.05:
            top_diseases.append({
                "disease": label_encoder.inverse_transform([idx])[0],
                "confidence": round(float(prob) * 100, 1)
            })

    # Primary prediction
    primary_disease = label_encoder.inverse_transform([prediction])[0]
    primary_confidence = round(float(probabilities[prediction]) * 100, 1)

    # Calculate severity from symptoms
    severity_score = calculate_severity_score(symptoms_input)

    # Calculate risk
    risk_score = calculate_risk(primary_disease, primary_confidence, severity_score)

    # Get precautions
    precautions = get_disease_precautions(primary_disease)

    # Build advice text
    advice = build_advice(primary_disease, precautions, risk_score)

    return {
        "primary_disease": primary_disease,
        "confidence": primary_confidence,
        "top_diseases": top_diseases,
        "matched_symptoms": matched_symptoms,
        "risk_score": risk_score,
        "precautions": precautions,
        "advice": advice,
        "severity_score": severity_score
    }


