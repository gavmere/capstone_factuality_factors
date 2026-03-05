# LLM Evaluation Testing Harness

This directory contains a comprehensive testing harness for evaluating LLM outputs against ground truth data for factuality factors.

## Usage

Run the Streamlit application:
```bash
streamlit run evals/app.py
```

## Expected CSV Format

Your ground truth CSV should have the following columns:

### Required Columns:
- `headline` - Article headline (string)
- `body` or `content` - Article body text (string)
- `url` (optional) - Article URL (string)

### Ground Truth Factor Columns:

1. **Clickbait** (numeric, 0-100)
   - Score from 0-100 indicating clickbait level
   - 0 = not clickbait, 100 = very clickbait

2. **Political Affiliation** (categorical)
   - Values: `Democratic`, `Republican`, `Neutral`, `Other`

3. **Sensationalism** (numeric, 0-100)
   - Score from 0-100 indicating sensationalism level
   - 0 = not sensational, 100 = very sensational

4. **Sentiment Analysis** (categorical)
   - Values: `Positive`, `Negative`, `Neutral`

5. **Toxicity** (categorical)
   - Values: `Friendly`, `Neutral`, `Rude`, `Toxic`, `Super_Toxic`
   - 5-level scale indicating toxicity level

6. **Headline-Body-Relation** (numeric, 0-100)
   - Score from 0-100 indicating how related the headline is to the body
   - 0 = no relation, 100 = very strong relation

### Example CSV:

```csv
headline,body,url,Clickbait,Political Affiliation,Sensationalism,Sentiment Analysis,Toxicity,Headline-Body-Relation
"Example Headline","Article body text here...","https://example.com/article",20,Neutral,30,Neutral,Neutral,95
```

## Features

- **Parallel Execution**: Configurable number of parallel workers for faster evaluation
- **Comprehensive Logging**: All inputs, outputs, and comparisons are logged to CSV files
- **Metrics Calculation**: Per-factor and overall accuracy, precision, recall, F1, MAE, RMSE
- **Visualization**: Confusion matrices and metrics visualization (requires plotly)
- **Error Handling**: Graceful handling of API errors, continues evaluation even if some articles fail

## Logs

All evaluation runs are logged to `evals/logs/`:
- `evaluation_logs_YYYYMMDD_HHMMSS.csv` - Individual run logs
- `master_log.csv` - Combined log of all runs
- `metadata/` - Full prompts and parameters for each run

## Notes

- The system automatically normalizes numeric values (handles both 0-1 and 0-100 scales)
- Categorical values are normalized (e.g., "Democrat" → "Democratic")
- Missing values are handled gracefully
- The system compares LLM outputs to ground truth with configurable tolerance for numeric factors
      