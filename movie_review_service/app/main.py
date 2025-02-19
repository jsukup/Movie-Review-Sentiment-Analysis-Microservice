from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tortoise import Tortoise
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Tortoise ORM on startup
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()

    yield

    # Cleanup on shutdown
    await Tortoise.close_connections()


app = FastAPI(title="Movie Review Service", lifespan=lifespan)

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
    "connections": {"default": "postgres://postgres:postgres@db:5432/movie_reviews"},
    "apps": {
        "models": {
            "models": ["app.models.review", "aerich.models"],
            "default_connection": "default",
        },
    },
}
