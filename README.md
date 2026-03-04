# capstone_factuality_factors

Factuality Factor Predictor: models and agents for analyzing news headlines and body text (clickbait, sensationalism, political affiliation, sentiment, toxicity, headline–body relation). Includes a Streamlit demo, Google ADK multi-agent pipeline, and an evaluation harness.

---

## Documentation checklist

- **Installation instructions** — see [Installation](#installation-instructions) below.
- **Dependency list (with versions)** — see [Dependencies](#dependency-list-with-versions) and `requirements.txt`.
- **Environment setup instructions** — see [Environment setup](#environment-setup-instructions).
- **Dataset access instructions** — see [Dataset and data access](#dataset-and-data-access).
- **Exact commands to run experiments** — see [Commands to run experiments](#exact-commands-to-run-experiments).
- **Expected outputs** — see [Expected outputs](#expected-outputs).
- **Directory structure** — see [Directory structure](#directory-structure).

---

## Installation instructions

1. **Create a virtual environment** (recommended). The project was developed and tested with Python’s built-in [venv](https://docs.python.org/3/library/venv.html).

   - **Unix/macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install dependencies** (see [Dependency list](#dependency-list-with-versions)):
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** — see [Environment setup](#environment-setup-instructions).

---

## Dependency list (with versions)

All dependencies are pinned in **`requirements.txt`** at the project root. Install with:

```bash
pip install -r requirements.txt
```

Key packages (versions as in `requirements.txt`): `google-adk` (ADK CLI and agent runtime), `streamlit`, `pandas`, `python-dotenv`, `openrouter`, `tqdm`, and ML/API-related libraries used by the models and evaluator. The file may include environment-specific or editable installs; use the same Python version (e.g. 3.12) for reproducibility.

---

## Environment setup instructions

- **Project root:** Create a `.env` file in the **project root** with at least:
  ```bash
  OPENROUTER_API_KEY=your_openrouter_key
  AI_STUDIO_API_KEY=your_ai_studio_key
  ```
  Used by the Streamlit demo (`demo.py`) and the CLI evaluator (`run_eval_cli.py`).

- **Factuality agents:** For `adk run` / `adk web` / agent eval, create a **`.env` inside `FactualityAgents/`** with:
  ```bash
  OPENROUTER_API_KEY=your_openrouter_key
  AI_STUDIO_API_KEY=your_ai_studio_key
  GEMINI_API_KEY=your_gemini_key
  ```
  Do not commit `.env` files; they are listed in `.gitignore`.

---

## Dataset and data access

Evaluation and demos use ground-truth CSV files in **`evals/`**:

- **`evals/ground_truth.csv`** — used by the Streamlit eval UI and by `run_eval_cli.py`.
- **`evals/REAL_GROUND_TRUTH_WITH_TOXICITY.csv`** — used by `run_agent_eval.py` for the full agent pipeline eval.

**Clickbait:**  
The Clickbait model is trained using data from [this Kaggle dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset).

---

## Exact commands to run experiments

Run these from the **project root** with your venv activated.

**Streamlit demo (frontend):**
```bash
streamlit run demo.py
```

**Agent demos (Google ADK):**
```bash
# Interactive CLI — paste headline/body, get factuality report
adk run FactualityAgents

# Web UI (development only) — open http://localhost:8000, select FactualityAgents
adk web --port 8000

# API server (for programmatic access)
adk api_server
```

**Evaluations:**
```bash
# Eval UI: upload CSV, pick model, run eval, see metrics
streamlit run evals/app.py

# CLI: single-LLM eval on evals/ground_truth.csv (uses OPENROUTER_API_KEY from root .env)
python run_eval_cli.py

# Agent orchestrator eval on evals/REAL_GROUND_TRUTH_WITH_TOXICITY.csv
python run_agent_eval.py          # 1 article (sanity check)
python run_agent_eval.py 10       # first 10 articles
python run_agent_eval.py all      # full dataset
```

---

## Expected outputs

- **`streamlit run demo.py`** — Browser opens at `http://localhost:8501`. You can enter a headline and optional body; the UI runs each factor model and shows scores (e.g. clickbait probability, sentiment, toxicity). You can add example screenshots here later.

- **`adk run FactualityAgents`** — Terminal prompt for input. After you submit a headline and body, the orchestrator returns a consolidated JSON-style factuality report (scores/labels per factor).

- **`adk web --port 8000`** — Browser at `http://localhost:8000` with an agent dropdown; choose FactualityAgents and chat to get the same style of report.

- **`streamlit run evals/app.py`** — Eval dashboard: upload CSV, configure model/prompts, run evaluation; results include per-factor accuracy, confusion matrices, and logs. You can add example screenshots here later.

- **`python run_eval_cli.py`** — Terminal output: “EVALUATION RESULTS” block with total articles, successful/errors, overall accuracy, per-factor accuracy and MAE, and mean MAE / mean accuracy at the end.

- **`python run_agent_eval.py [n|all]`** — Progress bar over articles, then “AGENT ORCHESTRATOR EVALUATION RESULTS” with overall accuracy and per-factor accuracy/MAE.

---

## Directory structure

```
capstone_factuality_factors/
├── demo.py                 # Streamlit frontend for factor prediction
├── run_eval_cli.py         # CLI single-LLM evaluation
├── run_agent_eval.py       # Agent orchestrator evaluation script
├── requirements.txt        # Pinned Python dependencies
├── .env                    # API keys (root; not committed)
│
├── evals/                  # Evaluation harness
│   ├── app.py              # Streamlit eval UI
│   ├── evaluator.py        # LLM evaluation logic
│   ├── logger.py           # Run and result logging
│   ├── utils.py            # Metrics, CSV validation, factor names
│   ├── validate_csv.py     # Ground truth CSV validation
│   ├── ground_truth.csv    # Default ground truth (eval UI + CLI)
│   ├── REAL_GROUND_TRUTH_WITH_TOXICITY.csv  # Used by run_agent_eval.py
│   ├── logs/               # evaluation_logs_*.csv, master_log.csv, metadata/
│   └── README.md           # Eval harness usage and CSV format
│
├── FactualityAgents/       # Google ADK agent package
│   ├── agent.py            # root_agent and sub-agents
│   ├── prompts.py          # System prompts and rubrics
│   ├── tools.py            # Tools that call trained models
│   └── .env                # API keys for agents (not committed)
│
├── models/                 # Trained factor predictors (FactualityFactor subclasses)
│   ├── factuality_factor.py
│   ├── clickbait/
│   ├── headline_body_relation/
│   ├── political_affiliation/
│   ├── sensationalism/
│   ├── sentiment_analysis/
│   └── toxicity/
│
├── model_training_scripts/  # Notebooks/scripts used to train the models
│   ├── clickbait/
│   ├── political_affiliation/
│   ├── sensationalism/
│   ├── sentiment_analysis/
│   └── toxicity/
│
├── generative_models/      # LLM client (e.g. OpenRouter)
│   └── LLM.py
│
└── tests/                  # Test suite
```

---

## Running the Frontend Application

To start the interactive frontend and use the Factuality Factor Predictor UI, follow these steps:

1. **Ensure you are in your project directory and your virtual environment is activated.**
2. **Launch the Streamlit app by running:**
   ```bash
   streamlit run demo.py
   ```
3. This will open up your browser to the application's interface. If your browser does not open automatically, copy the local URL provided in the terminal (e.g., `http://localhost:8501`) and paste it into your browser.

> **Note:**  
> - Ensure your `.env` file with necessary API keys is present in the root directory before launching the app.
> - If you make changes to the source code while the Streamlit app is running, the app will automatically reload to reflect the updates.

## Running the Factuality Agents (via Google ADK)

This project uses [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) to orchestrate a multi-agent system for factuality analysis. The orchestrator agent (`factuality_root_agent`) coordinates specialized sub-agents for clickbait detection, headline-body relation, political affiliation, sensationalism, sentiment, and toxicity analysis.

### Prerequisites

1. **Install dependencies** (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```
   This includes `google-adk`, which provides the `adk` CLI tool.

2. **Configure environment variables.** The agents require API keys to function. Make sure you have a `.env` file inside the `FactualityAgents/` directory with the following keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   AI_STUDIO_API_KEY=your_ai_studio_key
   GEMINI_API_KEY=your_gemini_key
   ```

### Option 1: Run in the Terminal (CLI)

Use `adk run` to interact with the agent directly in your terminal:

```bash
adk run FactualityAgents
```

> **Note:** Run this command from the **project root directory** (the parent of `FactualityAgents/`). ADK discovers the agent package by folder name and looks for the `root_agent` variable inside `agent.py`.

You will get an interactive prompt where you can paste a news headline and body for analysis. The orchestrator will delegate to the appropriate sub-agents and return a consolidated factuality report.

### Option 2: Run with the Web UI

ADK provides a built-in web interface for testing and debugging agents:

```bash
adk web --port 8000
```

> **Note:** Like `adk run`, execute this from the **project root directory**.

Then open [http://localhost:8000](http://localhost:8000) in your browser. Select `FactualityAgents` from the agent dropdown in the upper-left corner, and start chatting with the agent.

> **Caution:** The ADK Web UI is intended for **development and debugging only** and should not be used in production.

### Option 3: Run as an API Server

To expose the agents as a RESTful API (useful for integrating with other services):

```bash
adk api_server
```

This starts a server you can send HTTP requests to programmatically.

For the agent package layout, see [Directory structure](#directory-structure) (FactualityAgents/).

> **Learn more:** For full documentation on ADK commands, deployment, and advanced configuration, see the [official ADK documentation](https://google.github.io/adk-docs/).

---

## Running the Evaluation Harness

The `evals/` directory contains a testing harness for evaluating LLM and agent outputs against ground truth data. There are three ways to run evaluations:

### Option 1: Streamlit Web UI

Launch the interactive evaluation dashboard:

```bash
streamlit run evals/app.py
```

This opens a browser UI where you can:
1. Upload a ground truth CSV file.
2. Select a model provider (Gemini or OpenRouter) and model name.
3. Configure prompts and model parameters (temperature, max tokens, etc.).
4. Run the evaluation and view per-factor accuracy, confusion matrices, and detailed results.

### Option 2: CLI Evaluation (Single LLM)

Run a quick headless evaluation from the command line using `run_eval_cli.py`:

```bash
python run_eval_cli.py
```

This evaluates the ground truth CSV (`evals/ground_truth.csv`) against a single LLM (defaults to `google/gemini-3-flash-preview` via OpenRouter) and prints per-factor accuracy and MAE to the terminal. Edit the variables at the top of the script to change the model, provider, or CSV path.

### Option 3: Agent Orchestrator Evaluation

Run the full multi-agent pipeline (the ADK orchestrator) against the ground truth dataset:

```bash
python run_agent_eval.py [limit]
```

- Pass a number to limit to the first N articles (e.g., `python run_agent_eval.py 10`).
- Pass `all` to evaluate every article in the dataset: `python run_agent_eval.py all`.
- With no argument, it defaults to evaluating 1 article (useful for a quick sanity check).

This uses the `factuality_root_agent` orchestrator to delegate to all sub-agents and compares their combined output against the ground truth.

### Ground Truth CSV Format

Your CSV should contain the following columns:

| Column | Type | Description |
|---|---|---|
| `headline` | string | Article headline |
| `body` or `content` | string | Article body text |
| `url` | string (optional) | Article URL |
| `Clickbait` | numeric (0-100) | Clickbait score |
| `Headline-Body-Relation` | numeric (0-100) | Headline-body alignment score |
| `Sensationalism` | numeric (0-100) | Sensationalism score |
| `Political Affiliation` | categorical | `Democratic`, `Republican`, `Neutral`, or `Other` |
| `Sentiment Analysis` | categorical | `Positive`, `Negative`, or `Neutral` |
| `Toxicity` | categorical | `Friendly`, `Neutral`, `Rude`, `Toxic`, or `Super_Toxic` |

### Evaluation Logs

All evaluation runs are logged to `evals/logs/`:
- `evaluation_logs_YYYYMMDD_HHMMSS.csv` -- Individual run logs
- `master_log.csv` -- Combined log of all runs
- `metadata/` -- Full prompts and parameters for each run

---

## How to Create a Factuality Factor Predictor

1. **Create a Class**  
   Create a new class that inherits from the `FactualityFactor` base class.

2. **Initialize Variables**  
   In the constructor (`__init__` method), initialize any necessary variables or load pre-trained models.  
   > **Important:** Do **not** perform model training inside the constructor. Your model should already be trained and simply loaded here for use.

3. **Implement the `probability` Method**  
   Define a `probability` method that takes a string (such as a headline or text) as input.  
   This method should return a dictionary with the format:  
   ```python
   {'class_label': probability_value}
   ```
   where `class_label` is the name of the class you are predicting, and `probability_value` is a float (e.g., `0.14`), representing the probability assigned to that class.


## Importing and Using a Factor

You can import any factor class directly from the `models` package. Example with the Clickbait predictor:

```python
from dotenv import load_dotenv
import os
from models.clickbait.clickbait import Clickbait

load_dotenv()  # loads OPENROUTER_API_KEY from .env

factor = Clickbait(os.getenv("OPENROUTER_API_KEY"))
result = factor.probability("You Won't Believe What Happens Next!")
print(result)  # {'0': 0.12, '1': 0.88}
```

Replace the import path with the factor you need (e.g., `models.sentiment_analysis.sentiment_analysis.SentimentAnalysis`). The `probability` method always returns a dictionary mapping class labels to probabilities.
