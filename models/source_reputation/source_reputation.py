from models.factuality_factor import FactualityFactor
from typing import Dict, Any
from models.source_reputation.source_model import SourceTrust, ArticleCredibility, extract_article_text

class SourceReputation(FactualityFactor):
    def __init__(self):
        super().__init__("Source Reputation", "Probability represents the credibility of the information source, ranging from low to high reputation.")
        self.domain_model = SourceTrust()
        self.article_model = ArticleCredibility()

    def probability(self, text: str) -> Dict[str, float]:
        # If input is a URL, use both domain and article models
        if text.startswith("http://") or text.startswith("https://"):
            article_text = extract_article_text(text)
            article_score = self.article_model.score(article_text)
            domain_score = self.domain_model.check_domain(text)
            combined = round(0.6 * article_score["trust_score"] + 0.4 * domain_score["score"], 3)
            return {"combined_reputation": combined}
        else:
            # If input is raw text, use only article model
            article_score = self.article_model.score(text)
            return {"trust_score": article_score["trust_score"]}

if __name__ == "__main__":
    # Quick local demo similar to the sentiment notebook examples
    sr = SourceReputation()

    print("\n1st Example (trustworthy URL):")
    trustworthy = "https://www.nytimes.com/2025/11/19/us/politics/comey-vindictive-prosecution-trump.html"
    print(sr.probability(trustworthy))

    print("\n2nd Example (untrustworthy URL):")
    untrustworthy = "https://www.infowars.com"
    print(sr.probability(untrustworthy))

    print("\n3rd Example (raw article text):")
    sample_text = (
        "This blog post makes sensational claims with no cited sources and uses inflammatory language "
        "to push a conspiracy narrative."
    )
    print(sr.probability(sample_text))