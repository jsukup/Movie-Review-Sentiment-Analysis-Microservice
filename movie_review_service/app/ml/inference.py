import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = self._load_config()
        self.labels = ["negative", "positive"]

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "models/best_model/model_config.json"
        )
        with open(config_path) as f:
            return json.load(f)

    async def load_model(self):
        if self.model is None:
            print(f"Loading model from Hugging Face Hub: {self.config['model_id']}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_id"]
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_type"])
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully")

    async def predict(self, text):
        if self.model is None:
            await self.load_model()

        # Tokenize and predict
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model(**inputs)
        probabilities = outputs.logits.softmax(dim=1)

        # Get prediction
        prediction = probabilities.argmax().item()
        confidence = probabilities.max().item()

        return {
            "sentiment": self.labels[prediction],
            "confidence": float(confidence),
        }
