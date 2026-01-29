from google import genai
from google.genai import types

def generate_final_article_content(prompt, article):
    return prompt + f'Title:{article["title"]}\nSource:{article["source"]}\nAuthor:{article["author"]}\nPublication Date:{article["publication_date"]}\nContent:{article["content"]}'

def generate(api_key, system_prompt, prompt, article, model="gemini-2.5-pro"):
    result = ''
    client = genai.Client(
        api_key=api_key,
    )
    model = model
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=generate_final_article_content(prompt, article)),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "Clickbait": genai.types.Schema(
                    type = genai.types.Type.NUMBER,
                ),
                "Headline-Body-Relation": genai.types.Schema(
                    type = genai.types.Type.NUMBER,
                ),
                "Party Affliation": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Republican", "Democrat", "Other"],
                ),
                "Sensationalism": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Sensational", "Non-Sensational"],
                ),
                "Sentiment Analysis": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Positive", "Negative", 'Neutral'],
                ),
                "Toxicity": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Friendly", "Neutral", "Rude", "Toxic", "Super_Toxic"],
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=system_prompt),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        result += chunk.text
    return result

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    API_key = os.getenv("AI_STUDIO_API_KEY")
    ARTICLE = {
        "title": "Trump 'retired' a database tracking the most expensive weather disasters. Now it's back — and finding over $100B in losses",
        "source": "CNN",
        "author": "Andrew Freedman",
        "publication_date": "2025-10-22 10:00:00",
        "content": '''The Billion-Dollar Weather and Climate Disasters Database, which the Trump administration"retired" in May, has relaunched outside of the government using the same methodology. In its first update at the new site, the database shows that the first six months of 2025 have been the most expensive first six months of any year since 1980. The Billion-Dollar Database tracks the financial costs of property and other infrastructure destroyed by extreme weather disasters in the United States, focusing on events that caused $1 billion or more in damages. So far, 2025 has racked up $101.4 billion in such losses. The climate research nonprofit Climate Centralnow hosts the databaseand makes this information available to insurers, policy makers, broadcast meteorologists and ordinary citizens. The database was rebuilt and will be maintained by its previous administrator Adam Smith, a former economist at the National Oceanic and Atmospheric Administration, the agency which used to host it. Smith found 14 billion-dollar disasters in the first half of this year, including the Los Angeles wildfires in January and a tornado outbreak across the central US in mid-March.More billion-dollar disasters are likely to be added to the list before 2025 is over. Without the database, the public would have no easy way to track the cost of extreme weather events, many of which are becoming more common and severe because of climate change. But climate change is not the sole reason the database shows an upward trend in both the number of billion-dollar disasters and the amount they cost. Population growth and an increase in the number of buildings in harm's way are the dominant factors, according to Smith. "Either way you look at it, the rise in damages relates to human activities and choices, and so you need to use information in context to better evaluate future choices," he said. The frequency of billion-dollar disasters has particularly increased in the last decade, Smith said, occurring nearly twice as often compared to the 30-year inflation-adjusted average. Between 1980 and 2024, there were nine such disasters on average each year. In the past five years, that annual average has jumped to 24. Therecord for a single yearwas 28 events in 2023. In the first six months of 2025, the list of billion-dollar disasters is mostly comprised of severe thunderstorms and tornado outbreaks, reflecting the conspicuous absence of landfalling hurricanes so far this season. However, the LA fires in January cost $61.2 billion, making them the costliest wildfires in US history, according to Climate Central. Climate Central hired Smith after the NOAA economist took early retirement this year, as part of the Trump administration's push to shrink the federal bureaucracy. NOAA's official list stopsat the end of2024. Climate Central's version picks up in 2025 and will continue from there. The relaunched list uses the same methodology as the old NOAA one, Smith said. It relies on data from insurance companies and other sources, some of which is proprietary, to tally up total losses. The decision to discontinue the database was due in part to Smith's exit from NOAA. It would have been a difficult task for the agency to continue without him, but it could have done it, he said. The choice to discontinue the database was in keeping with the administration's focus on cutting climate change datasets and programs across federal agencies. But there were calls for it to continue from multiple sectors. "This dataset was simply too important to stop being updated. Demand for its revival actually came from several aspects of industry and society, including decision makers in the insurance, reinsurance risk space, academia, Congress and local communities," Smith said.'''
    }

    System_Prompt = """
    You are a helpful assistant that analyzes news articles and provides scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity.

    You will be given an article and you will need to analyze it and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity.

    Each Factor will be scored as such:

    - ClickBait:
    - Evaluate the disparity between sensational promises made in headlines and the actual information delivered in the content.
    - Measure the density of emotionally charged, hyperbolic, or curiosity-gap keywords (e.g., "shocking," "you won't believe") in titles.
    - Identify formatting patterns designed specifically to maximize click-through rates rather than inform, such as misleading thumbnails or all-caps headers.
    Questions you should ask yourself is: **Is this text clickbait? Is this so sensational that it is trying to get
    you to click on it, or pull your attention to it via a headline or text
    segment that is angering, fear inspiring or extremely
    controversial?
    For Example:
    Headline: "You won't believe what happened next!"
    Body: "You won't believe what happened next!"
    Score: .85

    Headline: "This is the most shocking thing you'll ever read!"
    Body: "This is the most shocking thing you'll ever read!"
    Score: .9

    Headline: "The latest news on the stock market"
    Body: "The latest news on the stock market"
    Score: 0

    Headline: "The weather in New York City today"
    Body: "The weather in New York City today"
    Score: 0

    Headline-Body-Relation: a score from 0 to 1 where 0 is no relation and 1 is a very strong relation. The headline should be a direct summary of the article.
    Questions you should ask yourself is: **Does the title , agree, discuss, is unrelated to, or negate the body**"
    For Example: 
    Headline: "Trump's new policy will make America great again!"
    Body: "Trump's new policy will make America great again!"
    Score: 1

    Headline: "The stock market is crashing!"
    Body: "The new most popular dog toy is the squeaky ball!"
    Score: 0


    Party Affliation: Democrat, Republican, or Other - This is based on the content of the article weather the writing is leaning towards a certain party.
    Questions you should ask yourself is: **Measure the differential framing of similar actions when committed by opposing political figures to detect double standards.
    Analyze sentiment scores associated with varying political entities to detect consistent favoritism or hostility regardless of the specific news.
    Check for systematic underreporting of negative news related to favored political groups and simultaneous overreporting for opponents.**
    For Example:
    Headline: "The stock market is crashing!"
    Body: "The new most popular dog toy is the squeaky ball!"
    Score: Other

    Headline: "I believe in affordable healthcare for all!"
    Body: "I believe in affordable healthcare for all!"
    Score: Democrat

    Headline: "I believe in Donald Trump for President 2028"
    Body: "I believe in Donald Trump for President 2028"
    Score: Republican


    Sensationalism: Sensational, or Non-Sensational - This is based on the content of the article and the language used. Sensationalism could be described as the use of emotional language to entice the reader to read the article.
    Questions you should ask yourself is: **Quantify the use of "click-words" and superlatives (e.g., "unprecedented," "catastrophic," "miracle") in non-extraordinary reporting contexts.
    Compare the emotional intensity of the headline against the actual evidential weight provided in the body text to detect hype.
    Identify reliance on shocking, isolated anecdotal evidence to make broad generalizations where statistical data would be more appropriate.**
    For Example:
    Headline: "The stock market is crashing!"
    Body: "The stock market is crashing and you should sell your stocks immediately! Your family will starve if you don't!"
    Score: Sensational

    Sentiment Analysis: Positive, Negative - This is based on the overall sentiment of the article. A positive sentiment is when an article is more positive or uplifting in nature. A negative sentiment is when an article is more negative or somber in nature.
    Questions you should ask yourself is: **Utilize NLP to score overall text polarity (positive/negative/neutral) and intensity to detect emotionally manipulative framing.
    Identify sudden, unjustified shifts in sentiment trajectory within a text that may indicate a pivot from reporting to editorializing.
    Compare the sentiment of the content against the neutral baseline expected for the specific topic or event type.**
    For Example:
    Headline: "The stock market is crashing!"
    Body: "The stock market is crashing and you should sell your stocks immediately! Your family will starve if you don't!"
    Score: Negative

    Toxicity: Friendly, Neutral, Rude, Toxic, or Super Toxic – This is based on the presence and severity of toxic language in the text and the corresponding toxicity score. If the toxicity score is 0.8 or higher, the toxicity should be Super Toxic. If the toxicity score is between 0.6 and 0.8, the toxicity should be Toxic. If the toxicity score is between 0.4 and 0.6, the toxicity should be Rude. If the toxicity score is between 0.2 and 0.4, the toxicity should be Neutral. If the toxicity score is below 0.2, the toxicity should be Friendly.

    Questions you should ask yourself are: Review the language used in the text to identify insults, slurs, profanity, or dehumanizing expressions.
    Evaluate whether the tone is hostile, threatening, or intended to provoke anger or fear.
    Determine whether the toxic language is central to the message or incidental (e.g., quoted speech or reporting).
    Assess whether specific individuals or groups are targeted and the severity of that targeting.

    Text: “These people are disgusting parasites ruining everything.”
    Score: Super Toxic

    Text: “That argument is ridiculous and only an idiot would believe it.”
    Score: Rude

    Text: “The proposal has generated strong reactions from both supporters and critics.”
    Score: Friendly
    """

    Prompt = """
    Analyze the following article and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity.
    """
    result = generate(API_key, System_Prompt, Prompt, ARTICLE, model='gemini-2.5-pro')
    print(result)
