import streamlit as st
import json
import os
import threading
from dotenv import load_dotenv
from generative_models.LLM import generate
from models.clickbait.clickbait import Clickbait
from models.headline_body_relation.headline_body_relation import HeadlineBodyRelation
from models.political_affiliation.political_affiliation import PoliticalAffiliation
from models.sensationalism.sensationalism import Sensationalism
from models.sentiment_analysis.sentiment_analysis import Sentiment
from models.source_reputation.source_reputation import SourceReputation

load_dotenv()

def run_with_timeout(func, timeout_seconds=60, *args, **kwargs):
    """Run a function with a timeout using threading (works in all environments including Streamlit)"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
    if exception[0]:
        raise exception[0]
    return result[0]

# Initialize models at startup
@st.cache_resource
def initialize_models():
    """Initialize all models and cache them in memory"""
    models = {}
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Initialize models that don't need API keys
    try:
        models["political_affiliation"] = PoliticalAffiliation()
    except Exception as e:
        print(f"Failed to load Political Affiliation model: {str(e)}")
        models["political_affiliation"] = None
    
    try:
        models["sensationalism"] = Sensationalism()
    except Exception as e:
        print(f"Failed to load Sensationalism model: {str(e)}")
        models["sensationalism"] = None
    
    try:
        models["sentiment"] = Sentiment()
    except Exception as e:
        print(f"Failed to load Sentiment Analysis model: {str(e)}")
        models["sentiment"] = None
    
    try:
        models["source_reputation"] = SourceReputation()
    except Exception as e:
        print(f"Failed to load Source Reputation model: {str(e)}")
        models["source_reputation"] = None
    
    # Initialize models that need API keys
    if openrouter_key:
        try:
            models["clickbait"] = Clickbait(openrouter_key)
        except Exception as e:
            print(f"Failed to load Clickbait model: {str(e)}")
            models["clickbait"] = None
        
        try:
            models["headline_body_relation"] = HeadlineBodyRelation(openrouter_key)
        except Exception as e:
            print(f"Failed to load Headline-Body-Relation model: {str(e)}")
            models["headline_body_relation"] = None
    else:
        print("OPENROUTER_API_KEY not found. Models requiring API keys will not be available.")
        models["clickbait"] = None
        models["headline_body_relation"] = None
    
    return models

# Initialize models at startup
MODELS = initialize_models()

FACTORS = {
    "Clickbait": {
        "enabled": True,
        "uses_headline": False,
        "llm_key": "Clickbait"
    },
    "Headline-Body-Relation": {
        "enabled": True,
        "uses_headline": True,
        "llm_key": "Headline-Body-Relation"
    },
    "Political Affiliation": {
        "enabled": True,
        "uses_headline": False,
        "llm_key": "Party Affliation"
    },
    "Sensationalism": {
        "enabled": False,
        "uses_headline": True,
        "llm_key": "Sensationalism"
    },
    "Sentiment Analysis": {
        "enabled": True,
        "uses_headline": False,
        "llm_key": "Sentiment Analysis"
    },
    "Source Reputation": {
        "enabled": False,
        "uses_headline": False,
        "llm_key": "Source Reputation"
    }
}

SYSTEM_PROMPT = """
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

PROMPT = "Analyze the following article and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation."

def format_article_content(headline, body):
    return f"Title: {headline}\nContent: {body}"

def get_llm_predictions(headline, body):
    api_key = os.getenv("AI_STUDIO_API_KEY")
    if not api_key:
        st.error("AI_STUDIO_API_KEY not found in environment variables")
        return None
    
    article = {
        "title": headline,
        "source": "Unknown",
        "author": "Unknown",
        "publication_date": "Unknown",
        "content": body
    }
    
    try:
        result = generate(api_key, SYSTEM_PROMPT, PROMPT, article)
        return json.loads(result)
    except Exception as e:
        st.error(f"Error getting LLM predictions: {str(e)}")
        return None

def get_model_predictions(headline, body, url=None, progress_bar=None, status_text=None):
    """Get predictions from all models using pre-initialized models"""
    model_results = {}
    total_models = 6  # Clickbait, Headline-Body-Relation, Political Affiliation, Sensationalism, Sentiment, Source Reputation
    current_model = 0
    
    # Clickbait - uses headline
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Clickbait model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("clickbait") is not None:
            clickbait_probs = MODELS["clickbait"].probability(headline)
            # Get the probability of class 1 (clickbait) or use the highest probability
            if "1" in clickbait_probs:
                model_results["Clickbait"] = clickbait_probs["1"]
            else:
                model_results["Clickbait"] = max(clickbait_probs.values())
        else:
            model_results["Clickbait"] = None
    except Exception as e:
        st.warning(f"Error in Clickbait model: {str(e)}")
        model_results["Clickbait"] = None
    
    # Headline-Body-Relation - uses headline and body
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Headline-Body-Relation model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("headline_body_relation") is not None:
            hbr_result = MODELS["headline_body_relation"].probability(headline, body)
            model_results["Headline-Body-Relation"] = hbr_result.get("similarity", 0.0)
        else:
            model_results["Headline-Body-Relation"] = None
    except Exception as e:
        st.warning(f"Error in Headline-Body-Relation model: {str(e)}")
        model_results["Headline-Body-Relation"] = None
    
    # Political Affiliation - uses body
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Political Affiliation model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("political_affiliation") is not None:
            pol_aff_probs = MODELS["political_affiliation"].probability(body)
            # Get the highest probability class
            if pol_aff_probs:
                max_class = max(pol_aff_probs.items(), key=lambda x: x[1])
                model_results["Political Affiliation"] = max_class[0].capitalize()
        else:
            model_results["Political Affiliation"] = None
    except Exception as e:
        st.warning(f"Error in Political Affiliation model: {str(e)}")
        model_results["Political Affiliation"] = None
    
    # Sensationalism - uses headline and body
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Sensationalism model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("sensationalism") is not None:
            # Combine headline and body for sensationalism analysis
            sensationalism_text = f"{headline}. {body}"
            # Run with timeout to prevent hanging
            try:
                sensationalism_result = run_with_timeout(
                    MODELS["sensationalism"].probability,
                    30,
                    sensationalism_text
                )
                if "sensationalism" in sensationalism_result:
                    model_results["Sensationalism"] = sensationalism_result["sensationalism"]
                else:
                    model_results["Sensationalism"] = None
            except TimeoutError:
                st.warning("Sensationalism model timed out after 30 seconds.")
                model_results["Sensationalism"] = None
        else:
            model_results["Sensationalism"] = None
    except Exception as e:
        st.warning(f"Error in Sensationalism model: {str(e)}")
        model_results["Sensationalism"] = None
    
    # Sentiment Analysis - uses body
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Sentiment Analysis model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("sentiment") is not None:
            sentiment_probs = MODELS["sentiment"].probability(body)
            # Get the highest probability sentiment
            if sentiment_probs:
                max_sentiment = max(sentiment_probs.items(), key=lambda x: x[1])
                model_results["Sentiment Analysis"] = max_sentiment[0].capitalize()
        else:
            model_results["Sentiment Analysis"] = None
    except Exception as e:
        st.warning(f"Error in Sentiment Analysis model: {str(e)}")
        model_results["Sentiment Analysis"] = None
    
    # Source Reputation - uses URL (required)
    current_model += 1
    if progress_bar:
        progress_bar.progress(current_model / total_models)
    if status_text:
        status_text.text(f"Running Source Reputation model ({current_model}/{total_models})...")
    
    try:
        if MODELS.get("source_reputation") is not None:
            # Use URL (required)
            source_input = url.strip() if url else ""
            if source_input:
                # Run with timeout to prevent hanging
                try:
                    source_reputation_result = run_with_timeout(
                        MODELS["source_reputation"].probability,
                        30,
                        source_input
                    )
                    # Get the trust score or combined reputation
                    if "trust_score" in source_reputation_result:
                        model_results["Source Reputation"] = source_reputation_result["trust_score"]
                    elif "combined_reputation" in source_reputation_result:
                        model_results["Source Reputation"] = source_reputation_result["combined_reputation"]
                    else:
                        model_results["Source Reputation"] = None
                except TimeoutError:
                    st.warning("Source Reputation model timed out after 30 seconds.")
                    model_results["Source Reputation"] = None
            else:
                model_results["Source Reputation"] = None
        else:
            model_results["Source Reputation"] = None
    except Exception as e:
        st.warning(f"Error in Source Reputation model: {str(e)}")
        model_results["Source Reputation"] = None
    
    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("Model predictions complete!")
    
    return model_results

def display_results(llm_results, model_results):
    """Display results in a table format with LLM and Model columns"""
    if not llm_results and not model_results:
        return
    
    enabled_factors = {k: v for k, v in FACTORS.items() if v["enabled"]}
    
    # Create a table with columns
    st.subheader("Analysis Results")
    
    # Create data for the table
    table_data = []
    for factor_name, factor_config in enabled_factors.items():
        llm_key = factor_config["llm_key"]
        
        # Get LLM result
        llm_value = None
        if llm_results and llm_key in llm_results:
            llm_value = llm_results[llm_key]
            if isinstance(llm_value, (int, float)):
                llm_display = f"{llm_value:.4f}"
            else:
                llm_display = str(llm_value)
        else:
            llm_display = "N/A"
        
        # Get Model result
        model_value = model_results.get(factor_name)
        if model_value is not None:
            if isinstance(model_value, (int, float)):
                model_display = f"{model_value:.4f}"
            else:
                model_display = str(model_value)
        else:
            model_display = "N/A"
        
        table_data.append({
            "Factor": factor_name,
            "LLM Prediction": llm_display,
            "Model Prediction": model_display
        })
    
    # Display as a table
    if table_data:
        st.table(table_data)

st.title("Factuality Factors Analysis")

# Show model initialization status in sidebar
with st.sidebar:
    st.header("Model Status")
    status_items = []
    if MODELS.get("political_affiliation") is not None:
        status_items.append("✓ Political Affiliation")
    if MODELS.get("sensationalism") is not None:
        status_items.append("✓ Sensationalism")
    if MODELS.get("clickbait") is not None:
        status_items.append("✓ Clickbait")
    if MODELS.get("headline_body_relation") is not None:
        status_items.append("✓ Headline-Body-Relation")
    if MODELS.get("sentiment") is not None:
        status_items.append("✓ Sentiment Analysis")
    if MODELS.get("source_reputation") is not None:
        status_items.append("✓ Source Reputation")
    
    if status_items:
        for item in status_items:
            st.success(item)
    else:
        st.warning("No models loaded. Check your API keys and model files.")

headline = st.text_area("Headline", height=100, placeholder="Enter the article headline...", key="headline_input")
body = st.text_area("Body", height=300, placeholder="Enter the article body content...", key="body_input")
url = st.text_input("URL (required, for Source Reputation)", placeholder="Enter article URL (e.g., https://example.com/article)", key="url_input")

if st.button("Analyze", key="analyze_button"):
    if not headline or not body or not url:
        st.warning("Please provide headline, body content, and URL.")
    else:
        # Create progress containers
        overall_progress = st.progress(0)
        status_text = st.empty()
        model_progress = None
        model_status = None
        
        try:
            # Get LLM predictions
            status_text.text("Getting LLM predictions...")
            overall_progress.progress(0.1)
            llm_results = get_llm_predictions(headline, body)
            overall_progress.progress(0.3)
            
            # Get model predictions with progress
            status_text.text("Running models...")
            model_progress = st.progress(0)
            model_status = st.empty()
            model_results = get_model_predictions(headline, body, url, model_progress, model_status)
            overall_progress.progress(0.9)
            
            # Complete
            overall_progress.progress(1.0)
            status_text.text("Analysis complete!")
            
            if llm_results or model_results:
                st.success("Analysis complete!")
                display_results(llm_results, model_results)
            else:
                st.error("Failed to get predictions.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # Keep progress bars visible to see where it failed