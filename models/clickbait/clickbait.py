from models.factuality_factor import FactualityFactor
from typing import Dict
import xgboost as xgb
from openai import OpenAI
import os

class Clickbait(FactualityFactor):
    def __init__(self, API_key: str):
        super().__init__("Clickbait", "Probability is defined as the likelihood that a headline is overly exaggerated or sensational to attract attention")
        self.API_key = API_key
        self.model = xgb.XGBClassifier()
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "clickbait_model.json")
        self.model.load_model(model_path)

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
        return {str(cls): prob for cls, prob in enumerate(class_probs)}

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    API_key = os.getenv("OPENROUTER_API_KEY")
    clickbait = Clickbait(API_key)
    print(clickbait.probability("You Won't Believe What Happens Next!"))
