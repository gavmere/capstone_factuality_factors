# capstone_factuality_factors

## Installation Guide

Follow these steps to set up your development environment and initialize the models required for the Factuality Factor Predictor project.

### 1. Create a Virtual Environment

This project was developed and tested using built-in [virtual environment](https://docs.python.org/3/library/venv.html) it is recommended to use a venv aswell

So create a venv such as:

- **Unix/macOS:**
  ```bash
  python3 -m venv [name_of_venv]
  source [name_of_venv]/bin/activate # To activate the venv
  ```
- **Windows:**
  ```bash
  python -m venv [name_of_venv]
  [name_of_venv]\Scripts\activate # To activate the venv
  ```

### 2. Install Package Dependencies

Ensure your virtual environment is activated, then install all required packages using the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Some functionality (such as LLM-based features or embeddings) requires API keys and environment variables. Create a `.env` file in your project root and add the necessary keys. For example:

```
OPENROUTER_API_KEY=your_api_key_here
AI_STUDIO_API_KEY=your_other_api_key
```

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

### Project Structure (Agent Package)

```
FactualityAgents/
  __init__.py      # Imports the agent module
  agent.py         # Defines root_agent (orchestrator) and all sub-agents
  prompts.py       # System prompts and rubrics for each sub-agent
  tools.py         # Tool functions that call the trained predictive models
  .env             # API keys (not committed to version control)
```

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
