from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_scores(sentence: str):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict
