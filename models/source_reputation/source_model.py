import requests
import tldextract
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SourceTrust:
    WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    CATEGORY_MAP = {
        "fake news": "fake",
        "conspiracy theory": "problematic",
        "pseudoscience": "problematic",
        "propaganda": "problematic",
        "satire": "neutral",
        "newspaper": "trustworthy",
        "news website": "trustworthy",
        "news agency": "trustworthy",
    }

    def normalize_domain(self, domain):
        ext = tldextract.extract(domain)
        return f"{ext.domain}.{ext.suffix}"

    def fetch_wikipedia_summary(self, name):
        try:
            url = self.WIKI_API.format(name)
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                return r.json().get("extract", "").lower()
        except:
            pass
        return ""

    def classify_from_wikipedia(self, domain):
        query = domain.replace(".", "_")
        text = self.fetch_wikipedia_summary(query)
        if not text:
            return "unknown"
        for kw, cat in self.CATEGORY_MAP.items():
            if kw in text:
                return cat
        return "neutral"

    def check_domain(self, domain):
        domain = self.normalize_domain(domain)
        category = self.classify_from_wikipedia(domain)
        score_map = {
            "trustworthy": 0.9,
            "neutral": 0.6,
            "unknown": 0.5,
            "problematic": 0.2,
            "fake": 0.1
        }
        return {
            "domain": domain,
            "category": category,
            "score": score_map.get(category, 0.5)
        }

class ArticleCredibility:
    MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
    def score(self, text: str):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        label = "real" if real_prob >= fake_prob else "fake"
        return {
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "label": label,
            "trust_score": real_prob
        }

def extract_article_text(url: str):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(paragraphs)
