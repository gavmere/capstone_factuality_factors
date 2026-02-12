import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from tools import (
    clickbait_score,
    headline_body_relation_score,
    political_affiliation_score,
    sensationalism_score,
    sentiment_score,
    toxicity_score,
)

load_dotenv()
NRP_PROXY_API_BASE = "https://ellm.nrp-nautilus.io/v1"
NRP_PROXY_API_KEY = os.getenv("NRP_PROXY_API_KEY")

MODEL_NAME = "openai/kimi"

MODEL = LiteLlm(model=MODEL_NAME, api_base=NRP_PROXY_API_BASE, api_key=NRP_PROXY_API_KEY)


class HeadlineInput(BaseModel):
    headline: str = Field(description="Headline text to analyze.")


class HeadlineBodyInput(BaseModel):
    headline: str = Field(description="Headline text to analyze.")
    body: str = Field(description="Article body text to compare.")


class TextInput(BaseModel):
    text: str = Field(description="Article or passage text to analyze.")

clickbait_agent = LlmAgent(
    name="clickbait_agent",
    model=MODEL,
    description="Scores clickbait likelihood for a headline.",
    instruction=(
        "You score clickbait likelihood for a headline. "
        "Call the clickbait_score tool with the given headline and return the tool result as JSON."
    ),
    tools=[clickbait_score],
    input_schema=HeadlineInput,
)

headline_body_agent = LlmAgent(
    name="headline_body_agent",
    model=MODEL,
    description="Scores headline/body semantic similarity.",
    instruction=(
        "You score headline/body relation. "
        "Call the headline_body_relation_score tool with the given headline and body, "
        "then return the tool result as JSON."
    ),
    tools=[headline_body_relation_score],
    input_schema=HeadlineBodyInput,
)

political_affiliation_agent = LlmAgent(
    name="political_affiliation_agent",
    model=MODEL,
    description="Scores political affiliation probabilities.",
    instruction=(
        "You score political affiliation. "
        "Call the political_affiliation_score tool with the given text and return the tool result as JSON."
    ),
    tools=[political_affiliation_score],
    input_schema=TextInput,
)

sensationalism_agent = LlmAgent(
    name="sensationalism_agent",
    model=MODEL,
    description="Scores sensationalism in text.",
    instruction=(
        "You score sensationalism. "
        "Call the sensationalism_score tool with the given text and return the tool result as JSON."
    ),
    tools=[sensationalism_score],
    input_schema=TextInput,
)

sentiment_agent = LlmAgent(
    name="sentiment_agent",
    model=MODEL,
    description="Scores sentiment in text.",
    instruction=(
        "You score sentiment. "
        "Call the sentiment_score tool with the given text and return the tool result as JSON."
    ),
    tools=[sentiment_score],
    input_schema=TextInput,
)

toxicity_agent = LlmAgent(
    name="toxicity_agent",
    model=MODEL,
    description="Scores toxicity in text.",
    instruction=(
        "You score toxicity. "
        "Call the toxicity_score tool with the given text and return the tool result as JSON."
    ),
    tools=[toxicity_score],
    input_schema=TextInput,
)

root_agent = LlmAgent(
    name="factuality_root_agent",
    model=MODEL,
    instruction=(
        "You are a factuality analysis assistant. "
        "Always call the appropriate sub-agent tool when the user asks for a factor "
        "or provides text/headline. Do not ask follow-up questions if the required "
        "input is present. If a headline is provided and the user asks about clickbait, "
        "call clickbait_agent with {\"headline\": ...}. If headline+body are provided, "
        "call headline_body_agent with {\"headline\": ..., \"body\": ...}. For any "
        "text-based factors (political affiliation, sensationalism, sentiment, toxicity), "
        "call the corresponding agent with {\"text\": ...}. "
        "If the user provides a quoted string and asks about clickbait, treat the "
        "quoted string as the headline. "
        "If multiple factors are requested, call each agent and return a single JSON "
        "object that maps factor name to the agent's result.\n\n"
        "Examples:\n"
        "User: Analyze if this is clickbait \"You will not believe this\"\n"
        "Action: call clickbait_agent({\"headline\": \"You will not believe this\"})\n"
        "User: Headline: \"X\" Body: \"Y\" analyze headline-body relation\n"
        "Action: call headline_body_agent({\"headline\": \"X\", \"body\": \"Y\"})"
    ),
    tools=[
        AgentTool(clickbait_agent),
        AgentTool(headline_body_agent),
        AgentTool(political_affiliation_agent),
        AgentTool(sensationalism_agent),
        AgentTool(sentiment_agent),
        AgentTool(toxicity_agent),
    ],
)

