from fastapi import APIRouter, HTTPException
from models.schemas import DrugCheckRequest, DrugCheckResponse
from services.drug_service import check_drug_interactions

router = APIRouter(prefix="/api", tags=["Drugs"])


@router.post("/drug-check", response_model=DrugCheckResponse)
async def drug_check(request: DrugCheckRequest):
    try:
        result = await check_drug_interactions(
            medicines=request.medicines,
            language=request.language
        )
        return DrugCheckResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drug check failed: {str(e)}"
        )

