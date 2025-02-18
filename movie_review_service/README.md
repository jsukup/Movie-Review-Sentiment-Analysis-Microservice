# Movie Review Service

A microservice for managing movie reviews with sentiment analysis capabilities.

## Features

- CRUD operations for movie reviews
- Movie review sentiment analysis (placeholder)
- PostgreSQL database with Tortoise ORM
- FastAPI-based REST API
- Docker containerization

## API Endpoints

- `POST /api/reviews` - Create a new review
- `GET /api/reviews/{movie_id}` - Get all reviews for a movie
- `PUT /api/reviews/{review_id}` - Update a review
- `DELETE /api/reviews/{review_id}` - Delete a review
- `POST /api/analyze` - Analyze review sentiment (placeholder)

## Setup

1. Clone the repository
2. Make sure you have Docker and Docker Compose installed
3. Run the following commands:

```bash
# Build and start the containers
docker-compose up --build

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## Development

The project uses:
- Python 3.10.12
- FastAPI
- Tortoise ORM
- PostgreSQL
- Docker

## Project Structure

```
/movie_review_service
    /app
        /api         - API routes and endpoints
        /models      - Database models
        /services    - Business logic
        /ml          - Machine learning components
    /tests          - Test files (to be added)
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name

## License

MIT 