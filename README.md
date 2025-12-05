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
