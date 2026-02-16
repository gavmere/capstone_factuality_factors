# Prompts for Factuality Factors Agents

# Configuration
USE_XML_PROMPTS = True


def wrap_xml(content: str, tag: str) -> str:
    return f"<{tag}>\n{content}\n</tag>" if USE_XML_PROMPTS else content


# --- CLICKBAIT ---
CLICKBAIT_RUBRIC = """
Score the headline from 0 to 100 based on its clickbait likelihood.
- 0-20: Factual, straightforward, and informative.
- 21-50: Slightly curiosity-driven but mostly representative of the content.
- 51-80: Strong clickbait characteristics (all-caps, hyperbolic, "you won't believe").
- 81-100: Pure clickbait, deceptive or extremely sensationalized.
"""

CLICKBAIT_EXAMPLES = """
Example 1:
Headline: "The stock market closed up 100 points today."
Score: 5
Reasoning: Purely factual reporting.

Example 2:
Headline: "This Shocking Secret About Apple Will Blow Your Mind!"
Score: 95
Reasoning: Classic clickbait pattern using hyperbolic language and a curiosity gap.
"""


def get_clickbait_prompt():
    prompt = (
        "You are an expert at detecting clickbait headlines.\n\n"
        "STEPS:\n"
        "1. Call the `clickbait_predictive_score` tool with the headline to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the headline using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{CLICKBAIT_RUBRIC}\n\n"
        f"EXAMPLES:\n{CLICKBAIT_EXAMPLES}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- HEADLINE-BODY RELATION ---
HBR_RUBRIC = """
Score the relation from 0 to 100.
- 90-100: Headline is a direct and accurate summary of the body.
- 50-80: Headline captures the main theme but misses details or emphasizes a secondary point.
- 10-40: Headline is only tangentially related to the content.
- 0: Headline is unrelated or contradicts the body.
"""


def get_hbr_prompt():
    prompt = (
        "Analyze the relationship between the headline and the body content.\n\n"
        "STEPS:\n"
        "1. Call the `headline_body_relation_predictive_score` tool with the headline and body to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the relationship using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{HBR_RUBRIC}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- POLITICAL AFFILIATION ---
POLITICAL_RUBRIC = """
Classify the political leaning of the text.
- Democratic: Leans towards progressive, liberal, or Democratic Party policies/views.
- Republican: Leans towards conservative, traditional, or Republican Party policies/views.
- Neutral: Balanced reporting with no clear partisan bias.
- Other: Partisan leaning that doesn't fit the US two-party system.
"""


def get_political_prompt():
    prompt = (
        "Identify the political affiliation of the provided text.\n\n"
        "STEPS:\n"
        "1. Call the `political_affiliation_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{POLITICAL_RUBRIC}\n\n"
        "Choose exactly one label: Democratic, Republican, Neutral, or Other. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- SENSATIONALISM ---
SENSATIONALISM_RUBRIC = """
Score sensationalism from 0 to 100.
- 0-30: Objective, neutral tone, focus on facts.
- 31-60: Moderate use of emotional language or emphasis.
- 61-100: Highly sensationalized, uses fear, anger, or extreme excitement to drive engagement.
"""


def get_sensationalism_prompt():
    prompt = (
        "Assess the level of sensationalism in the text.\n\n"
        "STEPS:\n"
        "1. Call the `sensationalism_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive score and your generative score (use is_numeric=True).\n\n"
        f"RUBRIC:\n{SENSATIONALISM_RUBRIC}\n\n"
        "Return a JSON object with 'predictive_score', 'generative_score', and 'final_score'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- SENTIMENT ---
SENTIMENT_RUBRIC = """
Classify the overall sentiment.
- Positive: Uplifting, successful, or optimistic tone.
- Negative: Critical, somber, or pessimistic tone.
- Neutral: Factual, dispassionate, or objective tone.
"""


def get_sentiment_prompt():
    prompt = (
        "Analyze the sentiment of the text.\n\n"
        "STEPS:\n"
        "1. Call the `sentiment_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{SENTIMENT_RUBRIC}\n\n"
        "Choose exactly one label: Positive, Negative, or Neutral. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- TOXICITY ---
TOXICITY_RUBRIC = """
Classify the toxicity level of the text.
- Friendly: Respectful, polite, and constructive.
- Neutral: Matter-of-fact, no toxic elements.
- Rude: Mildly disrespectful or dismissive.
- Toxic: Aggressive, insulting, or hostile.
- Super_Toxic: Hate speech, threats, or extreme dehumanization.
"""


def get_toxicity_prompt():
    prompt = (
        "Analyze the toxicity of the text.\n\n"
        "STEPS:\n"
        "1. Call the `toxicity_predictive_score` tool to get the statistical model's prediction.\n"
        "2. Perform your own analysis of the text using the RUBRIC below.\n"
        "3. Use the `combine_scores` tool to merge the predictive label and your generative label (use is_numeric=False).\n\n"
        f"RUBRIC:\n{TOXICITY_RUBRIC}\n\n"
        "Choose exactly one label: Friendly, Neutral, Rude, Toxic, or Super_Toxic. Return JSON with 'predictive_label', 'generative_label', and 'final_label'."
    )
    if USE_XML_PROMPTS:
        return f"<system_instruction>\n{prompt}\n</system_instruction>"
    return prompt


# --- ORCHESTRATOR ---
ORCHESTRATOR_PROMPT = """
You are the Factuality Orchestrator Agent. 
Your job is to coordinate a suite of specialized sub-agents to analyze a news article.

INPUT: You will receive an article with a headline and body.
PROCESS:
1. Delegate analysis to the appropriate sub-agents for: Clickbait, Headline-Body-Relation, Political Affiliation, Sentiment, and Toxicity.
2. Collect the individual scores and labels.
3. Synthesize the results into a final cohesive factuality report.

OUTPUT: Provide a JSON object containing all scores AND a markdown summary report.
"""
