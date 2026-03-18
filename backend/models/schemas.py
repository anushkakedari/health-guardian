from pydantic import BaseModel
from typing import List, Optional

class SymptomRequest(BaseModel):
    symptoms: str
    language: str = "english"
    age: Optional[int] = None
    gender: Optional[str] = None

class SymptomResponse(BaseModel):
    possible_conditions: List[str]
    risk_score: str        # low / medium / high
    advice: str
    see_doctor_urgently: bool
    translated_advice: str  # in user's chosen language

class DrugCheckRequest(BaseModel):
    medicines: List[str]
    language: str = "english"

class DrugInteraction(BaseModel):
    drug1: str
    drug2: str
    severity: str          # mild / moderate / severe
    description: str

class DrugCheckResponse(BaseModel):
    interactions: List[DrugInteraction]
    overall_risk: str      # safe / caution / danger
    advice: str
    translated_advice: str