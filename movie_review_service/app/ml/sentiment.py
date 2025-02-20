from pydantic import BaseModel
from enum import Enum


class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: SentimentEnum
    confidence: float
