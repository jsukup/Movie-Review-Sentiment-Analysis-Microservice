import json
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    # Class constant for easy adjustment of 'neutral' threshold
    CONFIDENCE_THRESHOLD = 0.60

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = self._load_config()
        self.labels = ["negative", "positive"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully")

    async def predict(self, text):
        if self.model is None:
            await self.load_model()

        # Tokenize and predict
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs.logits.softmax(dim=1)

        # Get prediction and confidence
        confidence = probabilities.max().item()

        # Use class constant for threshold
        if confidence < self.CONFIDENCE_THRESHOLD:
            sentiment = "neutral"
        else:
            prediction = probabilities.argmax().item()
            sentiment = self.labels[prediction]

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
        }
