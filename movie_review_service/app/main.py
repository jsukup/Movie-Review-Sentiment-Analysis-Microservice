from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise
from app.api.routes import router

app = FastAPI(title="Movie Review Service")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Tortoise ORM configuration
TORTOISE_ORM = {
    "connections": {
        "default": "postgres://postgres:postgres@db:5432/movie_reviews"
    },
    "apps": {
        "models": {
            "models": ["app.models.review", "aerich.models"],
            "default_connection": "default",
        },
    },
}

register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,
    add_exception_handlers=True,
) 