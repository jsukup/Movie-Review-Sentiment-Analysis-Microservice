# Movie Review Service

A FastAPI-based microservice that provides movie review management with sentiment analysis capabilities.

## Key Features

- **Clean Architecture**: Organized in layers (API, Services, Models) for maintainability
- **Database Management**: PostgreSQL with Tortoise ORM and Aerich migrations
- **CRUD Operations**: Complete review management functionality
- **ML Integration**: Sentiment analysis using fine-tuned DistilBERT
- **API Documentation**: Available at `/docs` endpoint

## Technical Details

See the detailed documentation in [movie_review_service/README.md](movie_review_service/README.md)

## Jupyter Notebook for Model Fine-tuning

The sentiment analysis model can be fine-tuned using the provided Jupyter notebook (movie_review_service/ml/training/fine_tuning_walkthrough.ipynb).

Follow these steps to run the fine-tuning process:

1. Set up the Python environment:

    ```bash
    cd /root/freelance-labs
    python -m venv venv
    source venv/bin/activate
    ```

2. Install required packages:

    ```bash
    pip install jupyter torch transformers datasets pandas numpy matplotlib scikit-learn tqdm ipywidgets
    ```

3. Start Jupyter notebook server:

    ```bash
    jupyter notebook --ip 0.0.0.0 --allow-root
    ```

4. Open the provided URL in your browser (it will look like):

    <http://127.0.0.1:8888/?token=><some_token>

5. Navigate to:

    '''
    movie_review_service/ml/training/fine_tuning_walkthrough.ipynb
    '''

6. Run the notebook:
   - Execute each cell in sequence using Shift + Enter
   - The notebook will guide you through:
     - Data preparation and preprocessing
     - Model configuration and training
     - Evaluation and testing

The fine-tuned model will be saved and can be used by the movie review service for sentiment analysis.
