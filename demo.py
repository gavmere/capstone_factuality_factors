import streamlit as st
import json
import os
import re
import time
import threading
import asyncio
from dotenv import load_dotenv

from generative_models.llm import generate
from models.clickbait.clickbait import Clickbait
from models.headline_body_relation.headline_body_relation import HeadlineBodyRelation
from models.political_affiliation.political_affiliation import PoliticalAffiliation
from models.sensationalism.sensationalism import Sensationalism
from models.sentiment_analysis.sentiment_analysis import Sentiment
from models.toxicity.toxicity import Toxicity
from FactualityAgents.prompts import FINAL_VERACITY_PROMPT

load_dotenv()

st.set_page_config(
    page_title="Factuality Factors",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Utility: timeout wrapper
# ---------------------------------------------------------------------------
def run_with_timeout(func, timeout_seconds=60, *args, **kwargs):
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
        raise TimeoutError(
            f"Function {func.__name__} timed out after {timeout_seconds} seconds"
        )
    if exception[0]:
        raise exception[0]
    return result[0]


# ---------------------------------------------------------------------------
# Model initialization (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_models():
    models = {}
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    for name, cls in [
        ("political_affiliation", PoliticalAffiliation),
        ("sensationalism", Sensationalism),
        ("sentiment", Sentiment),
        ("toxicity", Toxicity),
    ]:
        try:
            models[name] = cls()
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            models[name] = None

    if openrouter_key:
        for name, cls in [
            ("clickbait", Clickbait),
            ("headline_body_relation", HeadlineBodyRelation),
        ]:
            try:
                models[name] = cls(openrouter_key)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                models[name] = None
    else:
        models["clickbait"] = None
        models["headline_body_relation"] = None

    return models


MODELS = initialize_models()

# ---------------------------------------------------------------------------
# LLM prompts (from demo.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful assistant that analyzes news articles and provides scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity.

You will be given an article and you will need to analyze it and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity.

Each Factor will be scored as such:

Clickbait: a score from 0 to 1 where 0 is not clickbait and 1 is very clickbait.

Headline-Body-Relation: a score from 0 to 1 where 0 is no relation and 1 is a very strong relation.

Party Affliation: Democrat, Republican, or Other - Based on the content leaning towards a certain party.

Sensationalism: Sensational, or Non-Sensational - Based on the use of emotional language.

Sentiment Analysis: Positive, Negative - Based on the overall sentiment of the article.

Toxicity: Friendly, Neutral, Rude, Toxic, or Super_Toxic – Based on the presence and severity of toxic language.
"""

PROMPT = "Analyze the following article and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Toxicity."

FACTORS = {
    "Clickbait": {"llm_key": "Clickbait"},
    "Headline-Body-Relation": {"llm_key": "Headline-Body-Relation"},
    "Political Affiliation": {"llm_key": "Party Affliation"},
    "Sensationalism": {"llm_key": "Sensationalism"},
    "Sentiment Analysis": {"llm_key": "Sentiment Analysis"},
    "Toxicity": {"llm_key": "Toxicity"},
}


# ---------------------------------------------------------------------------
# Pipeline 1 — Pure LLM Predictions
# ---------------------------------------------------------------------------
def parse_llm_json_response(raw_response):
    if raw_response is None:
        raise ValueError("LLM returned no response content.")
    response_text = str(raw_response).strip()
    if not response_text:
        raise ValueError("LLM returned an empty response.")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    if "```" in response_text:
        for part in response_text.split("```"):
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(response_text[start : end + 1])
    raise ValueError("LLM response did not contain valid JSON.")


def get_llm_predictions(headline, body):
    provider = "openrouter"
    model = os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.5-pro")
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        if provider == "openrouter"
        else os.getenv("AI_STUDIO_API_KEY")
    )
    if not api_key:
        return None
    article = {
        "title": headline,
        "source": "Unknown",
        "author": "Unknown",
        "publication_date": "Unknown",
        "content": body,
    }
    try:
        result = generate(
            api_key, SYSTEM_PROMPT, PROMPT, article, model=model, provider=provider
        )
        return parse_llm_json_response(result)
    except Exception as e:
        st.warning(f"LLM prediction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline 2 — Statistical Model Predictions
# ---------------------------------------------------------------------------
def get_model_predictions(headline, body):
    results = {}

    try:
        if MODELS.get("clickbait"):
            probs = MODELS["clickbait"].probability(headline)
            results["Clickbait"] = probs.get("1", max(probs.values()))
        else:
            results["Clickbait"] = None
    except Exception:
        results["Clickbait"] = None

    try:
        if MODELS.get("headline_body_relation"):
            hbr = MODELS["headline_body_relation"].probability(headline, body)
            results["Headline-Body-Relation"] = hbr.get("similarity", 0.0)
        else:
            results["Headline-Body-Relation"] = None
    except Exception:
        results["Headline-Body-Relation"] = None

    try:
        if MODELS.get("political_affiliation"):
            probs = MODELS["political_affiliation"].probability(body)
            if probs:
                label = max(probs.items(), key=lambda x: x[1])
                results["Political Affiliation"] = label[0].capitalize()
            else:
                results["Political Affiliation"] = None
        else:
            results["Political Affiliation"] = None
    except Exception:
        results["Political Affiliation"] = None

    try:
        if MODELS.get("sensationalism"):
            text = f"{headline}. {body}"
            res = run_with_timeout(MODELS["sensationalism"].probability, 30, text)
            results["Sensationalism"] = res.get("sensationalism") if res else None
        else:
            results["Sensationalism"] = None
    except Exception:
        results["Sensationalism"] = None

    try:
        if MODELS.get("sentiment"):
            probs = MODELS["sentiment"].probability(body)
            if probs:
                label = max(probs.items(), key=lambda x: x[1])
                results["Sentiment Analysis"] = label[0].capitalize()
            else:
                results["Sentiment Analysis"] = None
        else:
            results["Sentiment Analysis"] = None
    except Exception:
        results["Sentiment Analysis"] = None

    try:
        if MODELS.get("toxicity"):
            text = body.strip()
            if text:
                category = run_with_timeout(MODELS["toxicity"].categorize, 30, text)
                results["Toxicity"] = category
            else:
                results["Toxicity"] = None
        else:
            results["Toxicity"] = None
    except Exception:
        results["Toxicity"] = None

    return results


# ---------------------------------------------------------------------------
# Pipeline 3 — ADK Agent Orchestration
# ---------------------------------------------------------------------------
def parse_agent_response(events):
    """Extract structured JSON from the agent event stream."""
    from evals.utils import NUMERIC_FACTORS, CATEGORICAL_FACTORS, normalize_factor_name

    final_text = ""
    for event in reversed(events):
        if hasattr(event, "content") and event.content:
            if hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text = part.text
                        break
            if final_text:
                break
        if hasattr(event, "text") and event.text:
            final_text = event.text
            break
    if not final_text:
        return {}

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", final_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{.*\})", final_text, re.DOTALL)
    if not json_match:
        return {}

    try:
        data = json.loads(json_match.group(1))
    except Exception:
        return {}

    mapping = {
        "clickbait_score": "Clickbait",
        "clickbait": "Clickbait",
        "headline_body_alignment": "Headline-Body-Relation",
        "headline_body_relation": "Headline-Body-Relation",
        "political_affiliation": "Political Affiliation",
        "sensationalism_score": "Sensationalism",
        "sensationalism": "Sensationalism",
        "sentiment": "Sentiment Analysis",
        "toxicity": "Toxicity",
    }
    normalized = {}
    valid_factors = NUMERIC_FACTORS + CATEGORICAL_FACTORS
    for k, v in data.items():
        norm_key = mapping.get(k.lower(), normalize_factor_name(k))
        if norm_key in valid_factors:
            if isinstance(v, dict):
                for sk in ["final_score", "final_label", "score", "label"]:
                    if sk in v:
                        normalized[norm_key] = v[sk]
                        break
                else:
                    normalized[norm_key] = next(iter(v.values()))
            else:
                normalized[norm_key] = v
    return normalized


def run_agent_prediction(headline, body):
    """Run the ADK root_agent and return parsed results."""
    try:
        from google.adk import Runner
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from FactualityAgents.agent import root_agent
    except ImportError as e:
        return {"_error": f"Agent dependencies not available: {e}"}

    async def _run():
        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            session_service=session_service,
            app_name="FactualityComprehensiveApp",
        )
        input_text = f"Analyze this article:\nHeadline: {headline}\nBody: {body}"
        events = await runner.run_debug(
            input_text,
            session_id=f"app_{int(time.time())}",
            quiet=True,
        )
        return events

    try:
        loop = asyncio.new_event_loop()
        events = loop.run_until_complete(_run())
        loop.close()
        return parse_agent_response(events)
    except Exception as e:
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _coerce_percent_score(value):
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score <= 1:
        score *= 100
    return max(0.0, min(100.0, score))


def _get_pipeline_value(results, factor_name):
    if not results:
        return None
    if factor_name in FACTORS:
        return results.get(FACTORS[factor_name]["llm_key"])
    return results.get(factor_name)


def _majority_label(values):
    labels = [str(v) for v in values if v not in (None, "N/A", "")]
    if not labels:
        return None
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return _to_json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _build_final_veracity_inputs(llm_results, model_results, agent_results):
    numeric_factors = ["Clickbait", "Headline-Body-Relation", "Sensationalism"]
    fused = {}

    for factor_name in numeric_factors:
        values = [
            _coerce_percent_score(_get_pipeline_value(llm_results, factor_name)),
            _coerce_percent_score(_get_pipeline_value(model_results, factor_name)),
            _coerce_percent_score(_get_pipeline_value(agent_results, factor_name)),
        ]
        values = [value for value in values if value is not None]
        fused[factor_name] = round(sum(values) / len(values), 2) if values else None

    fused["Sentiment Analysis"] = _majority_label(
        [
            _get_pipeline_value(llm_results, "Sentiment Analysis"),
            _get_pipeline_value(model_results, "Sentiment Analysis"),
            _get_pipeline_value(agent_results, "Sentiment Analysis"),
        ]
    )
    fused["Toxicity"] = _majority_label(
        [
            _get_pipeline_value(llm_results, "Toxicity"),
            _get_pipeline_value(model_results, "Toxicity"),
            _get_pipeline_value(agent_results, "Toxicity"),
        ]
    )
    fused["Political Affiliation"] = _majority_label(
        [
            _get_pipeline_value(llm_results, "Political Affiliation"),
            _get_pipeline_value(model_results, "Political Affiliation"),
            _get_pipeline_value(agent_results, "Political Affiliation"),
        ]
    )

    return fused


def get_final_veracity_prediction(headline, body, llm_results, model_results, agent_results):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.warning("OPENROUTER_API_KEY is required for the final veracity pass.")
        return None

    fused = _build_final_veracity_inputs(llm_results, model_results, agent_results)
    replacements = {
        "{title}": headline,
        "{body}": body,
        "{clickbait_score}": str(
            fused["Clickbait"] if fused["Clickbait"] is not None else "N/A"
        ),
        "{headline_body_score}": str(
            fused["Headline-Body-Relation"]
            if fused["Headline-Body-Relation"] is not None
            else "N/A"
        ),
        "{sensationalism_score}": str(
            fused["Sensationalism"] if fused["Sensationalism"] is not None else "N/A"
        ),
        "{sentiment_label}": str(fused["Sentiment Analysis"] or "N/A"),
        "{toxicity_label}": str(fused["Toxicity"] or "N/A"),
        "{political_label}": str(fused["Political Affiliation"] or "N/A"),
    }
    final_prompt = FINAL_VERACITY_PROMPT
    for placeholder, value in replacements.items():
        final_prompt = final_prompt.replace(placeholder, value)

    raw_context = {
        "llm_results": _to_json_safe(llm_results),
        "statistical_results": _to_json_safe(model_results),
        "agent_results": _to_json_safe(agent_results),
        "fused_inputs": _to_json_safe(fused),
    }

    article = {
        "title": "",
        "source": "",
        "author": "",
        "publication_date": "",
        "content": "",
    }

    try:
        response = generate(
            api_key,
            "Return only valid JSON.",
            f"{final_prompt}\n\nAdditional pipeline context:\n{json.dumps(raw_context, indent=2)}",
            article,
            model=os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.5-pro"),
            provider="openrouter",
        )
        parsed = parse_llm_json_response(response)
        parsed["_fused_inputs"] = fused
        return parsed
    except Exception as e:
        st.warning(f"Final veracity prediction error: {e}")
        return None


# ===========================================================================
# App Layout
# ===========================================================================

# --- Sidebar ---
with st.sidebar:
    st.header("Model Status")
    model_labels = {
        "clickbait": "Clickbait",
        "headline_body_relation": "Headline-Body-Relation",
        "political_affiliation": "Political Affiliation",
        "sensationalism": "Sensationalism",
        "sentiment": "Sentiment Analysis",
        "toxicity": "Toxicity",
    }
    for key, label in model_labels.items():
        if MODELS.get(key) is not None:
            st.success(label)
        else:
            st.error(f"{label} — unavailable")

    st.divider()
    st.header("Pipelines")
    run_llm = st.checkbox("LLM Predictions", value=True)
    run_stat = st.checkbox("Statistical Models", value=True)
    run_agent = st.checkbox("Agent Orchestration", value=True)

    st.divider()
    st.caption(
        "Built with Streamlit · "
        "[Project Report](https://www.gavmere.me/GenAIForGood/)"
    )

# --- Ethos banner ---
st.title("Generative AI for Good")
st.info(
    "Combining **Generative AI**, **Traditional Machine Learning**, and "
    "**Human-in-the-Loop** evaluation to detect misinformation online. "
    "Rather than making a binary \"fake news\" determination, we assess articles on six "
    "distinct *factuality factors* — Clickbait, Headline-Body Relation, Political "
    "Affiliation, Sensationalism, Sentiment Analysis, and Toxicity — providing readers "
    "with actionable, multidimensional insights about article quality.  \n\n"
    "[Read the full technical report](https://www.gavmere.me/GenAIForGood/)",
    icon=":material/article:",
)

st.header("Factuality Factors Analysis")

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "final_veracity_results" not in st.session_state:
    st.session_state.final_veracity_results = None

# --- Input section ---
col_input, col_info = st.columns([3, 2], gap="large")

with col_input:
    headline = st.text_area(
        "Headline",
        height=80,
        placeholder="Enter the article headline...",
        key="headline_input",
    )
    body = st.text_area(
        "Body",
        height=220,
        placeholder="Enter the article body content...",
        key="body_input",
    )
    analyze_btn = st.button("Analyze Article", type="primary", use_container_width=True)

with col_info:
    st.subheader("How it works")
    st.markdown(
        "Each article is processed through **three independent pipelines**, "
        "and the results are shown side-by-side so you can compare:"
    )
    st.markdown(
        """
| Pipeline | What it does |
|---|---|
| **LLM** | A large language model scores every factor in a single prompt. |
| **Statistical** | Purpose-built ML models (XGBoost, Logistic Regression, VADER, RoBERTa) each tackle one factor. |
| **Agent** | A multi-agent system (Google ADK) delegates to specialized sub-agents that combine LLM reasoning with statistical model tool calls. |
"""
    )
    with st.expander("The six factuality factors"):
        st.markdown(
            """
- **Clickbait** — Does the headline use sensational language to bait clicks? *(0-1 score)*
- **Headline-Body Relation** — How well does the headline represent the content? *(0-1 score)*
- **Political Affiliation** — Does the article show partisan lean? *(Democrat / Republican / Other)*
- **Sensationalism** — Does it use emotional language to evoke strong reactions? *(0-1 score)*
- **Sentiment Analysis** — Overall emotional tone. *(Positive / Negative / Neutral)*
- **Toxicity** — Hostile or offensive language severity. *(Friendly to Super_Toxic)*
"""
        )

# --- Analysis ---
if analyze_btn:
    if not headline or not body:
        st.warning("Please provide both a headline and body text.")
    else:
        llm_results = None
        model_results = None
        agent_results = None

        progress = st.progress(0, text="Starting analysis...")
        step = 0
        total_steps = sum([run_llm, run_stat, run_agent])
        if total_steps == 0:
            st.info("Select at least one pipeline in the sidebar.")
        else:
            if run_llm:
                progress.progress(step / total_steps, text="Running LLM predictions...")
                with st.spinner("Querying LLM..."):
                    llm_results = get_llm_predictions(headline, body)
                step += 1
                progress.progress(step / total_steps, text="LLM complete.")

            if run_stat:
                progress.progress(step / total_steps, text="Running statistical models...")
                with st.spinner("Running statistical models..."):
                    model_results = get_model_predictions(headline, body)
                step += 1
                progress.progress(step / total_steps, text="Statistical models complete.")

            if run_agent:
                progress.progress(
                    step / total_steps, text="Running agent orchestration..."
                )
                with st.spinner("Running agent orchestration..."):
                    agent_results = run_agent_prediction(headline, body)
                    if agent_results and "_error" in agent_results:
                        st.warning(f"Agent error: {agent_results['_error']}")
                        agent_results = None
                step += 1
                progress.progress(step / total_steps, text="Agent complete.")

            progress.progress(1.0, text="All pipelines finished.")
            st.session_state.analysis_results = {
                "headline": headline,
                "body": body,
                "llm_results": llm_results,
                "model_results": model_results,
                "agent_results": agent_results,
            }
            st.session_state.final_veracity_results = None

analysis_results = st.session_state.analysis_results

if analysis_results:
    llm_results = analysis_results["llm_results"]
    model_results = analysis_results["model_results"]
    agent_results = analysis_results["agent_results"]

    st.divider()
    st.header("Results")

    active = [
        (n, r)
        for n, r in [
            ("LLM", llm_results),
            ("Statistical", model_results),
            ("Agent", agent_results),
        ]
        if r
    ]
    if not active:
        st.error("No pipeline returned results. Check your API keys and model status.")
    else:
        st.subheader("Side-by-Side Comparison")

        table_data = []
        for factor_name, cfg in FACTORS.items():
            row = {"Factor": factor_name}
            llm_key = cfg["llm_key"]
            row["LLM"] = _fmt(llm_results.get(llm_key) if llm_results else None)
            row["Statistical"] = _fmt(
                model_results.get(factor_name) if model_results else None
            )
            row["Agent"] = _fmt(agent_results.get(factor_name) if agent_results else None)
            table_data.append(row)

        st.table(table_data)

        tabs = st.tabs([name for name, _ in active])
        for tab, (name, results) in zip(tabs, active):
            with tab:
                cols = st.columns(3)
                factor_items = list(FACTORS.items())
                for i, (factor_name, cfg) in enumerate(factor_items):
                    key = cfg["llm_key"] if name == "LLM" else factor_name
                    val = results.get(key)
                    with cols[i % 3]:
                        st.metric(label=factor_name, value=_fmt(val))

        with st.expander("Raw JSON responses"):
            raw_cols = st.columns(len(active))
            for col, (name, results) in zip(raw_cols, active):
                with col:
                    st.markdown(f"**{name}**")
                    st.json(results)

        st.subheader("Final Veracity")
        st.caption(
            "Run a second-stage audit using `FINAL_VERACITY_PROMPT`, combining the prior pipeline outputs into a single final veracity score."
        )
        if st.button("Run Final Veracity Audit", use_container_width=True):
            with st.spinner("Running final veracity audit..."):
                st.session_state.final_veracity_results = get_final_veracity_prediction(
                    analysis_results["headline"],
                    analysis_results["body"],
                    llm_results,
                    model_results,
                    agent_results,
                )

        if st.session_state.final_veracity_results:
            final_veracity = st.session_state.final_veracity_results
            fused_inputs = final_veracity.get("_fused_inputs", {})
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Final Score", _fmt(final_veracity.get("final_score")))
            with metric_cols[1]:
                st.metric("Veracity Label", _fmt(final_veracity.get("veracity_label")))

            if final_veracity.get("reasoning_summary"):
                st.markdown("**Reasoning Summary**")
                st.write(final_veracity["reasoning_summary"])

            with st.expander("Fused inputs sent to FINAL_VERACITY_PROMPT"):
                st.json(fused_inputs)

            with st.expander("Final veracity raw JSON"):
                raw_output = dict(final_veracity)
                raw_output.pop("_fused_inputs", None)
                st.json(raw_output)
