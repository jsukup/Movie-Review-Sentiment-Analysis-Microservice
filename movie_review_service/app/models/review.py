from tortoise import fields
from tortoise.models import Model


class Review(Model):
    id = fields.IntField(pk=True)
    movie_id = fields.IntField()
    content = fields.TextField()
    rating = fields.IntField()
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "reviews"

    def __str__(self):
        return f"Review(id={self.id}, movie_id={self.movie_id}, rating={self.rating})" 