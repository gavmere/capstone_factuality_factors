import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from .tools import (
    clickbait_predictive_score,
    headline_body_relation_predictive_score,
    political_affiliation_predictive_score,
    sensationalism_predictive_score,
    sentiment_predictive_score,
    toxicity_predictive_score,
    combine_scores,
    final_veracity_scoring_agent,
)
from .prompts import (
    get_clickbait_prompt,
    get_hbr_prompt,
    get_political_prompt,
    get_sensationalism_prompt,
    get_sentiment_prompt,
    get_toxicity_prompt,
    ORCHESTRATOR_PROMPT,
)

load_dotenv()

# --- OpenRouter Model Setup ---
MODEL = LiteLlm(
    model="openrouter/google/gemini-3-flash-preview",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
)


# --- Input Schemas ---
class ArticleInput(BaseModel):
    headline: str = Field(description="Article headline")
    body: str = Field(description="Article body content")


class TextInput(BaseModel):
    text: str = Field(description="Text content to analyze")


class HeadlineInput(BaseModel):
    headline: str = Field(description="Headline text")


# --- Specialized Agents (Hybrid Scoring) ---

clickbait_agent = LlmAgent(
    name="clickbait_agent",
    model=MODEL,
    instruction=get_clickbait_prompt(),
    tools=[clickbait_predictive_score, combine_scores],
    input_schema=HeadlineInput,
)

headline_body_agent = LlmAgent(
    name="headline_body_agent",
    model=MODEL,
    instruction=get_hbr_prompt(),
    tools=[headline_body_relation_predictive_score, combine_scores],
    input_schema=ArticleInput,
)

political_affiliation_agent = LlmAgent(
    name="political_affiliation_agent",
    model=MODEL,
    instruction=get_political_prompt(),
    tools=[political_affiliation_predictive_score, combine_scores],
    input_schema=TextInput,
)

sensationalism_agent = LlmAgent(
    name="sensationalism_agent",
    model=MODEL,
    instruction=get_sensationalism_prompt(),
    tools=[sensationalism_predictive_score, combine_scores],
    input_schema=TextInput,
)

sentiment_agent = LlmAgent(
    name="sentiment_agent",
    model=MODEL,
    instruction=get_sentiment_prompt(),
    tools=[sentiment_predictive_score, combine_scores],
    input_schema=TextInput,
)

toxicity_agent = LlmAgent(
    name="toxicity_agent",
    model=MODEL,
    instruction=get_toxicity_prompt(),
    tools=[toxicity_predictive_score, combine_scores],
    input_schema=TextInput,
)

# --- Orchestrator Agent ---

root_agent = LlmAgent(
    name="factuality_root_agent",
    model=MODEL,
    instruction=ORCHESTRATOR_PROMPT,
    tools=[
        AgentTool(clickbait_agent),
        AgentTool(headline_body_agent),
        AgentTool(political_affiliation_agent),
        AgentTool(sensationalism_agent),
        AgentTool(sentiment_agent),
        AgentTool(toxicity_agent),
        final_veracity_scoring_agent,
    ],
)
