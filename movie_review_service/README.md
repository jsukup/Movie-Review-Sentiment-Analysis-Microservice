# Movie Review Service

A microservice for managing movie reviews with sentiment analysis capabilities.

## Features

- CRUD operations for movie reviews
- Sentiment analysis using fine-tuned DistilBERT model
- PostgreSQL database with Tortoise ORM and Aerich migrations
- FastAPI-based REST API
- Docker containerization

## Technical Implementation

### 1. Code Quality & Project Structure
```
/movie_review_service
    /app
        /api         - API routes and endpoints
        /models      - Database models and schemas
        /services    - Business logic layer
        /ml          - Machine learning components
            /models  - Trained model artifacts
    /migrations      - Database migration files
    /ml
        /training   - Model training scripts
    /tests          - Test files
```

### 2. ORM & Database Management
- **ORM**: Tortoise ORM for async database operations (**SCALABILITY!**)
- **Migrations**: Aerich for version-controlled database schema changes (not implemented yet)
- **Models**: Defined in `app/models` with proper relationships and validation

### 3. CRUD Operations
- `POST /api/reviews` - Create new review
- `GET /api/reviews/{movie_id}` - Retrieve reviews by movie
- `PUT /api/reviews/{review_id}` - Update existing review
- `DELETE /api/reviews/{review_id}` - Delete review

### 4. Sentiment Analysis Implementation
#### Endpoint
- `POST /api/analyze` - Analyzes review sentiment
- Implements model caching for improved performance
- Async implementation for better scalability
- Error handling for robustness

#### Model Architecture
- Base: DistilBERT (lightweight BERT variant)
- Fine-tuned on IMDB Dataset (from Kaggle)
- Binary classification (positive/negative sentiment)
- Training/Validation split: 80/20 with stratified sampling
- Evaluation metrics: Accuracy, Precision, Recall, F1-score (Accuracy used for determining "best model")

## Setup

1. Clone the repository
2. Install Docker and Docker Compose
3. Run the following commands:

```bash
# Build and start the containers
docker-compose up --build

# Initialize database migrations
docker-compose exec api aerich init -t app.main.TORTOISE_ORM
docker-compose exec api aerich init-db

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Database Migrations
For future schema changes:
```bash
# Create a new migration
docker-compose exec api aerich migrate --name <migration_name>

# Apply migrations
docker-compose exec api aerich upgrade
```

## Development

The project uses:
- Python 3.10.12
- FastAPI with async endpoints
- Tortoise ORM + Aerich migrations
- PostgreSQL
- HuggingFace Transformers
- Docker

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name

## License
(C) 2025, All rights reserved. John Sukup and Expected X, LLC.