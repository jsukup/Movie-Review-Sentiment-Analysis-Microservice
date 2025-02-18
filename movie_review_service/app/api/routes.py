from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.services.review_service import ReviewService
from app.ml.sentiment import SentimentAnalyzer


router = APIRouter()


class ReviewCreate(BaseModel):
    movie_id: int
    content: str
    rating: int


class ReviewUpdate(BaseModel):
    content: str
    rating: int


class ReviewResponse(BaseModel):
    id: int
    movie_id: int
    content: str
    rating: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.post("/reviews", response_model=ReviewResponse)
async def create_review(review: ReviewCreate):
    return await ReviewService.create_review(
        movie_id=review.movie_id, content=review.content, rating=review.rating
    )


@router.get("/reviews/{movie_id}", response_model=List[ReviewResponse])
async def get_reviews(movie_id: int):
    return await ReviewService.get_reviews_by_movie(movie_id)


@router.put("/reviews/{review_id}", response_model=ReviewResponse)
async def update_review(review_id: int, review: ReviewUpdate):
    updated_review = await ReviewService.update_review(
        review_id=review_id, content=review.content, rating=review.rating
    )
    if not updated_review:
        raise HTTPException(status_code=404, detail="Review not found")
    return updated_review


@router.delete("/reviews/{review_id}")
async def delete_review(review_id: int):
    deleted = await ReviewService.delete_review(review_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"message": "Review deleted successfully"}


@router.post("/analyze")
async def analyze_sentiment(text: str):
    sentiment = SentimentAnalyzer.analyze(text)
    return {"sentiment": sentiment}
