from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
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


class Review(ReviewCreate):
    id: int
    created_at: str
    updated_at: str


@router.post("/reviews", response_model=Review)
async def create_review(review: ReviewCreate):
    return await ReviewService.create_review(
        movie_id=review.movie_id,
        content=review.content,
        rating=review.rating
    )


@router.get("/reviews/{movie_id}", response_model=List[Review])
async def get_reviews(movie_id: int):
    return await ReviewService.get_reviews_by_movie(movie_id)


@router.put("/reviews/{review_id}", response_model=Review)
async def update_review(review_id: int, review: ReviewUpdate):
    updated_review = await ReviewService.update_review(
        review_id=review_id,
        content=review.content,
        rating=review.rating
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