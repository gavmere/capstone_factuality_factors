from models.factuality_factor import FactualityFactor
from typing import Dict
from openai import OpenAI
import numpy as np
import os

class HeadlineBodyRelation(FactualityFactor):
    def __init__(self, API_key: str):
        super().__init__("Headline Body Relation", "Gets the cosine similarity between the headline and body of the article")
        self.API_key = API_key

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

    def probability(self, headline: str, body: str) -> Dict[str, float]:
        headline_embedding = self.get_embedding(headline)
        body_embedding = self.get_embedding(body)
        max_chunk_words = 5000
        body_words = body.split()
        if len(body_words) > max_chunk_words:
            body_chunks = [
                " ".join(body_words[i:i+max_chunk_words])
                for i in range(0, len(body_words), max_chunk_words)
            ]
        else:
            body_chunks = [body]
        chunk_embeddings = [self.get_embedding(chunk) for chunk in body_chunks]
        body_embedding = np.mean(np.array(chunk_embeddings), axis=0)
        similarity = np.dot(headline_embedding, body_embedding) / (np.linalg.norm(headline_embedding) * np.linalg.norm(body_embedding))
        return {'similarity': similarity}


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    API_key = os.getenv("OPENROUTER_API_KEY")
    headline_body_relation = HeadlineBodyRelation(API_key)
    headline = "Ice cream is now no longer cold"
    body = (
        "In a startling culinary shift, ice cream across the globe has mysteriously lost its chill. "
        "Consumers and scientists alike are baffled by pints and cones that remain creamy and sweet, "
        "but never cold to the touch. “I bought two tubs last night and both were room temperature the whole time,” "
        "reported local resident Maya Sanchez. Ice cream parlors everywhere are scrambling to adjust, offering "
        "novel ‘warm sundaes’ and ‘lukewarm milkshakes.’ Meteorologists note that this phenomenon transcends weather, "
        "with reports flowing in from the Arctic Circle to equatorial regions. Food safety experts assure the public "
        "that the products are still safe to eat, though many agree that the summer treat just isn’t the same. "
        "Meanwhile, freezer manufacturers are running tests, and some suspect a yet-undiscovered property of the "
        "flavors themselves. Only time will tell if frozen desserts will ever be truly cold again."
    )

    print(f"Headline: {headline}\n\nArticle body: {body}")
    print(headline_body_relation.probability(headline, body))