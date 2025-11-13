from google import genai
from google.genai import types

def generate_final_article_content(prompt, article):
    return prompt + f'Title:{article["title"]}\nSource:{article["source"]}\nAuthor:{article["author"]}\nPublication Date:{article["publication_date"]}\nContent:{article["content"]}'

def generate(api_key, system_prompt, prompt, article, model="gemini-2.0-flash"):
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
                "Source Reputation": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Credible", "Non-Credible", "Caution"],
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
    You are a helpful assistant that analyzes news articles and provides scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation.

    You will be given an article and you will need to analyze it and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation.

    Each Factor will be scored as such:

    Clickbait: a score from 0 to 1 where 0 is not clickbait and 1 is very clickbait. CLickbait could be described as a headline that is overly dramatic or sensational or uses emotional language to entice the reader to click on the article.
    An example of clickbait is "You won't believe what happened next!" or "This is the most shocking thing you'll ever read!" not clickbait would be something like "The latest news on the stock market" or "The weather in New York City today"

    Headline-Body-Relation: a score from 0 to 1 where 0 is no relation and 1 is a very strong relation. The headline should be a direct summary of the article.
    For Example: 
    Headline: "Trump's new policy will make America great again!"
    Body: "Trump's new policy will make America great again!"
    Score: 1

    Headline: "The stock market is crashing!"
    Body: "The new most popular dog toy is the squeaky ball!"
    Score: 0

    Party Affliation: Democrat, Republican, or Other - This is based on the content of the article weather the writing is leaning towards a certain party.
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
    For Example:
    Headline: "The stock market is crashing!"
    Body: "The stock market is crashing and you should sell your stocks immediately! Your family will starve if you don't!"
    Score: Sensational

    Sentiment Analysis: Positive, Negative - This is based on the overall sentiment of the article. A positive sentiment is when an article is more positive or uplifting in nature. A negative sentiment is when an article is more negative or somber in nature.
    For Example:
    Headline: "The stock market is crashing!"
    Body: "The stock market is crashing and you should sell your stocks immediately! Your family will starve if you don't!"
    Score: Negative

    Source Reputation: Credible, Non-Credible, or Caution - This is based on the reputation of the source and the credibility of the information. If the source is a known fake news source, then the source reputation should be Non-Credible. If the source is a known reliable source, then the source reputation should be Credible. If the source is a known mixed source, then the source reputation should be Caution.
    Source: CNN
    Score: Credible

    Source: The Donald Trump News Network
    Score: Non-Credible

    Source: The New York Times
    Score: Credible

    Source: The Onion
    Score: Non-Credible
    """

    Prompt = """
    Analyze the following article and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation.
    """
    result = generate(API_key, System_Prompt, Prompt, ARTICLE, model='gemini-2.5-pro')
    print(result)
