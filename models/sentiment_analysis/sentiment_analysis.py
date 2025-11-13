from models.factuality_factor import FactualityFactor
from typing import Dict
import xgboost as xgb
from openai import OpenAI
import os
import joblib

class Sentiment(FactualityFactor):
    def __init__(self, API_key: str):
        super().__init__(
            "Sentiment",
            "Probability represents the emotional tone of the text: positive, neutral, or negative."
        )
        self.API_key = API_key
        self.model = xgb.XGBClassifier()
        model_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(model_dir, "sentiment.json")
        gz_path = os.path.join(model_dir, "sentiment_model.gz")
        
        json_exists = os.path.exists(json_path)
        if json_exists:
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(json_path)
            except Exception:
                os.remove(json_path)
                json_exists = False
        
        if not json_exists:
            if os.path.exists(gz_path):
                self.model = joblib.load(gz_path)
                self.model.save_model(json_path)
            else:
                raise FileNotFoundError(f"Neither {json_path} nor {gz_path} found")

        # Map class indices to sentiment labels
        self.class_map = {0: "negative", 1: "neutral", 2: "positive"}

    def get_embedding(self, text: str) -> list[float]:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.API_key,
        )
        embedding = client.embeddings.create(
            model="google/gemini-embedding-001",
            input=text,
            encoding_format="float"
        )
        return embedding.data[0].embedding

    def probability(self, text: str) -> Dict[str, float]:
        embedding = self.get_embedding(text)
        class_probs = self.model.predict_proba([embedding])[0]
        return {self.class_map[i]: float(prob) for i, prob in enumerate(class_probs)}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    API_key = os.getenv("OPENROUTER_API_KEY")

    sentiment = Sentiment(API_key)
    print(sentiment.probability("I hate you this is the worst thing ever"))
    print(sentiment.probability("I love you this is the best thing ever"))
