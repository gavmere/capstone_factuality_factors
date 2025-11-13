from models.clickbait.clickbait import Clickbait
from models.headline_body_relation.headline_body_relation import HeadlineBodyRelation
from models.political_affiliation.political_affiliation import PoliticalAffiliation
from models.source_reputation.source_reputation import SourceReputation
from models.sentiment_analysis.sentiment_analysis import SentimentAnalysis
from models.sesationalism.sensationalism import Sensationalism
from dotenv import load_dotenv
import os

load_dotenv()
API_key = os.getenv("OPENROUTER_API_KEY")

def test_clickbait():
    print("Testing Clickbait...")
    factor = Clickbait(API_key)
    print(factor.get_name())
    print(factor.get_description())
    test_str_true = 'You Won\'t Believe What Happens Next!'
    test_str_false = 'This is a test article about political clickbait.'
    print('Clickbait probability:')
    print(f'{test_str_true}: {factor.probability(test_str_true)}')
    print(f'{test_str_false}: {factor.probability(test_str_false)}')

def test_headline_body_relation():
    print("Testing Headline Body Relation...")
    factor = HeadlineBodyRelation(API_key)
    print(factor.get_name())
    print(factor.get_description())
    test_headline_true = 'Ice cream is now no longer cold'
    test_body_true = (
        "In a startling culinary shift, ice cream across the globe has mysteriously lost its chill. "
        "Consumers and scientists alike are baffled by pints and cones that remain creamy and sweet, but never cold to the touch. "
        '"I bought two tubs last night and both were room temperature the whole time," reported local resident Maya Sanchez. '
        "Ice cream parlors everywhere are scrambling to adjust, offering novel 'warm sundaes' and 'lukewarm milkshakes.' "
        "Meteorologists note that this phenomenon transcends weather, with reports flowing in from the Arctic Circle to equatorial regions. "
        "Food safety experts assure the public that the products are still safe to eat, though many agree that the summer treat just isn't the same. "
        "Meanwhile, freezer manufacturers are running tests, and some suspect a yet-undiscovered property of the flavors themselves. "
        "Only time will tell if frozen desserts will ever be truly cold again."
    )
    test_headline_false = (
        "This headline does not relate to the body at all."
    )
    test_body_false = (
        "In a startling culinary shift, ice cream across the globe has mysteriously lost its chill. "
        "Consumers and scientists alike are baffled by pints and cones that remain creamy and sweet, but never cold to the touch. "
        '"I bought two tubs last night and both were room temperature the whole time," reported local resident Maya Sanchez. '
        "Ice cream parlors everywhere are scrambling to adjust, offering novel 'warm sundaes' and 'lukewarm milkshakes.' "
        "Meteorologists note that this phenomenon transcends weather, with reports flowing in from the Arctic Circle to equatorial regions. "
        "Food safety experts assure the public that the products are still safe to eat, though many agree that the summer treat just isn't the same. "
        "Meanwhile, freezer manufacturers are running tests, and some suspect a yet-undiscovered property of the flavors themselves. "
        "Only time will tell if frozen desserts will ever be truly cold again."
    )
    print('Headline Body Relation probability:')
    print(f'{test_headline_true}: {factor.probability(test_headline_true, test_body_true)}')
    print(f'{test_headline_false}: {factor.probability(test_headline_false, test_body_false)}')

def test_political_affiliation():
    print("Testing Political Affiliation...")
    factor = PoliticalAffiliation()
    print(factor.get_name())
    print(factor.get_description())
    print(factor.probability("This is a test article about political clickbait."))

def test_source_reputation():
    print("Testing Source Reputation...")
    factor = SourceReputation()
    print(factor.get_name())
    print(factor.get_description())
    print(factor.probability("This is a test article source with clickbait."))

def test_sentiment_analysis():
    print("Testing Sentiment Analysis...")
    factor = SentimentAnalysis(API_key)
    print(factor.get_name())
    print(factor.get_description())
    print(factor.probability("This is a clickbait article with sentiment."))

def test_sensationalism():
    print("Testing Sensationalism...")
    factor = Sensationalism()
    print(factor.get_name())
    print(factor.get_description())
    print(factor.probability("This is a sensational clickbait test article."))

if __name__ == "__main__":
    test_clickbait()
    test_headline_body_relation()
    test_political_affiliation()
    test_source_reputation()
    test_sentiment_analysis()
    test_sensationalism()