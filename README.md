# capstone_factuality_factors

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

## Running the Built-in Tests

1. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your `.env` file (in the project root) contains a valid `OPENROUTER_API_KEY` value. The tests exercise the clickbait factor, which makes embedding requests.
3. From the project root directory, run:
   ```bash
   python -m tests
   ```
   This executes the smoke tests defined in `tests.py` for each factor class.

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
