from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from movie_review_service.app.ml.inference import SentimentAnalyzer


# Define request model
class SentimentRequest(BaseModel):
    text: str


# Define response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


router = APIRouter()
analyzer = SentimentAnalyzer()


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = await analyzer.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
