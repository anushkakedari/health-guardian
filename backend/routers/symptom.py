from fastapi import APIRouter, HTTPException
from models.schemas import SymptomRequest, SymptomResponse
from services.ml_service import predict_disease, get_all_symptoms
from services.translation_service import translate_text, translate_list

router = APIRouter(prefix="/api", tags=["Symptoms"])


@router.post("/analyze", response_model=SymptomResponse)
async def analyze(request: SymptomRequest):
    try:
        # Step 1 — Split symptoms by comma and clean
        symptoms_list = [s.strip() for s in request.symptoms.split(",")]

        # Step 2 — ML model predicts disease + risk + advice
        ml_result = predict_disease(symptoms_list)

        # Step 3 — Translate advice to user's language
        translated_advice = translate_text(
            ml_result["advice"],
            request.language
        )

        # Step 4 — Translate precautions to user's language
        translated_precautions = translate_list(
            ml_result["precautions"],
            request.language
        )

        # Step 5 — Build and return response
        return SymptomResponse(
            possible_conditions=[
                d["disease"] for d in ml_result["top_diseases"]
            ],
            risk_score=ml_result["risk_score"],
            advice=ml_result["advice"],
            see_doctor_urgently=ml_result["risk_score"] == "high",
            translated_advice=translated_advice
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/symptoms-list")
def symptoms_list():
    """Returns all symptoms the ML model understands"""
    return {"symptoms": get_all_symptoms()}

