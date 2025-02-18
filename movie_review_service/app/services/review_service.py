from typing import List, Optional
from app.models.review import Review


class ReviewService:
    @staticmethod
    async def create_review(movie_id: int, content: str, rating: int) -> Review:
        return await Review.create(
            movie_id=movie_id,
            content=content,
            rating=rating
        )

    @staticmethod
    async def get_reviews_by_movie(movie_id: int) -> List[Review]:
        return await Review.filter(movie_id=movie_id).all()

    @staticmethod
    async def get_review_by_id(review_id: int) -> Optional[Review]:
        return await Review.filter(id=review_id).first()

    @staticmethod
    async def update_review(review_id: int, content: str, rating: int) -> Optional[Review]:
        review = await Review.filter(id=review_id).first()
        if review:
            review.content = content
            review.rating = rating
            await review.save()
        return review

    @staticmethod
    async def delete_review(review_id: int) -> bool:
        deleted_count = await Review.filter(id=review_id).delete()
        return deleted_count > 0 