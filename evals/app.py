"""
Streamlit application for LLM evaluation testing harness.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np

# Optional plotly import for visualizations
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from evals.evaluator import Evaluator
from evals.logger import EvaluationLogger
from evals.utils import (
    validate_csv_structure,
    NUMERIC_FACTORS,
    CATEGORICAL_FACTORS,
    get_body_column
)
# Load environment variables
load_dotenv()

# Default system prompt (from LLM.py)
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that analyzes news articles and provides scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation.

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
Score: Non-Credible"""

DEFAULT_USER_PROMPT = "Analyze the following article and provide scores for the following factors: Clickbait, Headline-Body-Relation, Party Affliation, Sensationalism, Sentiment Analysis, Source Reputation."

# Available models by provider
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]

OPENROUTER_MODELS = [
    # OpenAI
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-3.5-turbo",
    # Anthropic
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-5-sonnet-20241022",
    # Google
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    # Meta
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3-70b-instruct",
    # Mistral
    "mistralai/mistral-large-2407",
    "mistralai/mixtral-8x7b-instruct",
    # Cohere
    "cohere/command-r-plus",
    "cohere/command-r",
    # Other popular models
    "perplexity/llama-3.1-sonar-large-128k-online",
    "qwen/qwen-2.5-72b-instruct",
    "01-ai/yi-1.5-34b-chat",
]

ALL_MODELS = {
    "Gemini": GEMINI_MODELS,
    "OpenRouter": OPENROUTER_MODELS
}

# Initialize logger
logger = EvaluationLogger()

st.set_page_config(
    page_title="LLM Evaluation Testing Harness",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Evaluation Testing Harness")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Run Evaluation", "View Logs", "Past Runs"]
)

if page == "Run Evaluation":
    st.header("Run Evaluation")
    
    # File Upload Section
    st.subheader("1. Upload Ground Truth CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV file with ground truth data",
        type=["csv"],
        help="CSV should contain: headline, body/content, and ground truth columns for factors"
    )
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} articles")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(df.head(10))
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Validate structure
            is_valid, missing = validate_csv_structure(df)
            if not is_valid:
                st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
            else:
                st.success("✅ CSV structure is valid")
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            df = None
    
    if df is not None:
        # Model Configuration Section
        st.subheader("2. Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            provider = st.selectbox(
                "Provider",
                ["Gemini", "OpenRouter"],
                index=0,
                help="Choose between Google Gemini or OpenRouter"
            )
        
        with col2:
            if provider == "Gemini":
                default_model = "gemini-2.5-pro"
                default_key = os.getenv("AI_STUDIO_API_KEY", "")
                key_env_var = "AI_STUDIO_API_KEY"
                model_help = "Enter a Gemini model name (e.g., gemini-2.5-pro, gemini-1.5-pro)"
            else:
                default_model = "openai/gpt-4o"
                default_key = os.getenv("OPENROUTER_API_KEY", "")
                key_env_var = "OPENROUTER_API_KEY"
                model_help = "Enter an OpenRouter model name (e.g., openai/gpt-4o, anthropic/claude-3.5-sonnet)"
            
            model = st.text_input(
                "Model",
                value=default_model,
                help=model_help
            )
            
            # Show suggested models
            with st.expander("💡 Suggested Models"):
                if provider == "Gemini":
                    st.write("**Popular Gemini models:**")
                    for m in GEMINI_MODELS:
                        st.code(m, language=None)
                else:
                    st.write("**Popular OpenRouter models:**")
                    st.write("**OpenAI:**")
                    for m in [m for m in OPENROUTER_MODELS if "openai" in m]:
                        st.code(m, language=None)
                    st.write("**Anthropic:**")
                    for m in [m for m in OPENROUTER_MODELS if "anthropic" in m]:
                        st.code(m, language=None)
                    st.write("**Other:**")
                    for m in [m for m in OPENROUTER_MODELS if "openai" not in m and "anthropic" not in m]:
                        st.code(m, language=None)
        
        with col3:
            api_key = st.text_input(
                "API Key",
                value=default_key,
                type="password",
                help=f"Leave empty to use {key_env_var} from .env"
            )
            if not api_key:
                if provider == "Gemini":
                    api_key = os.getenv("AI_STUDIO_API_KEY")
                else:
                    api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Prompt Configuration
        st.subheader("3. Prompt Configuration")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=300,
            help="System instruction for the LLM"
        )
        
        user_prompt = st.text_area(
            "User Prompt",
            value=DEFAULT_USER_PROMPT,
            height=100,
            help="User prompt template"
        )
        
        # Parameter Configuration
        st.subheader("4. Model Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Sampling temperature"
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=8192,
                value=2048,
                step=100,
                help="Maximum output tokens"
            )
        
        with col3:
            top_p = st.slider(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.05,
                help="Top-p sampling parameter"
            )
        
        with col4:
            top_k = st.number_input(
                "Top-k",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                help="Top-k sampling parameter (optional)"
            )
        
        # Execution Configuration
        st.subheader("5. Execution Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            import os
            default_workers = min(10, os.cpu_count() or 1)
            max_workers = st.number_input(
                "Parallel Workers",
                min_value=1,
                max_value=20,
                value=default_workers,
                help=f"Number of parallel workers (auto-detected: {default_workers})"
            )
        
        with col2:
            tolerance = st.slider(
                "Numeric Tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Tolerance for numeric factor comparisons"
            )
        
        # Run Evaluation
        st.subheader("6. Run Evaluation")
        
        if st.button("🚀 Start Evaluation", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Please provide an API key")
            else:
                # Initialize evaluator
                evaluator = Evaluator(
                    logger=logger,
                    tolerance=tolerance,
                    max_workers=max_workers
                )
                
                # Prepare parameters
                parameters = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                }
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(completed, total):
                    # Ensure progress doesn't exceed 1.0
                    progress = min(completed / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {completed}/{total} articles ({progress*100:.1f}%)")
                
                # Run evaluation
                try:
                    with st.spinner("Running evaluation..."):
                        # Determine provider from selection
                        provider_lower = provider.lower()
                        results = evaluator.evaluate_dataset(
                            df=df,
                            api_key=api_key,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            parameters=parameters,
                            provider=provider_lower,
                            progress_callback=progress_callback
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Evaluation complete!")
                    
                    # Store results in session state
                    st.session_state['last_results'] = results
                    st.session_state['run_id'] = results['run_id']
                    
                    st.success(f"✅ Evaluation complete! Run ID: {results['run_id']}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display Results
        if 'last_results' in st.session_state:
            st.subheader("7. Results")
            results = st.session_state['last_results']
            
            # Summary
            st.write("### Summary")
            summary = results['summary']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", summary['total_articles'])
            with col2:
                st.metric("Successful", summary['successful'])
            with col3:
                st.metric("Errors", summary['errors'])
            with col4:
                overall_acc = summary['metrics'].get('overall', {}).get('accuracy', 0)
                st.metric("Overall Accuracy", f"{overall_acc*100:.2f}%")
            
            # Metrics Table
            st.write("### Metrics by Factor")
            metrics_data = []
            for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
                factor_metrics = summary['metrics'].get(factor, {})
                metrics_data.append({
                    "Factor": factor,
                    "Accuracy": f"{factor_metrics.get('accuracy', 0)*100:.2f}%",
                    "Correct": factor_metrics.get('num_correct', 0),
                    "Total": factor_metrics.get('num_total', 0),
                    "MAE": f"{factor_metrics.get('mae', 'N/A')}" if factor in NUMERIC_FACTORS else "N/A",
                    "RMSE": f"{factor_metrics.get('rmse', 'N/A')}" if factor in NUMERIC_FACTORS else "N/A",
                    "F1": f"{factor_metrics.get('f1', 'N/A')}" if factor in CATEGORICAL_FACTORS else "N/A"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion Matrices
            st.write("### Confusion Matrices")
            for factor in CATEGORICAL_FACTORS:
                factor_metrics = summary['metrics'].get(factor, {})
                cm = factor_metrics.get('confusion_matrix')
                labels = factor_metrics.get('confusion_matrix_labels')
                
                if cm and labels:
                    st.write(f"**{factor}**")
                    cm_df = pd.DataFrame(
                        cm,
                        index=labels,
                        columns=labels
                    )
                    st.dataframe(cm_df)
                    
                    # Visual confusion matrix
                    if PLOTLY_AVAILABLE:
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual"),
                            x=labels,
                            y=labels,
                            text_auto=True,
                            aspect="auto",
                            title=f"Confusion Matrix: {factor}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("💡 Install plotly (pip install plotly) to see visual confusion matrices")
            
            # Detailed Results Table
            st.write("### Detailed Results")
            detailed_data = []
            for result in results['results'][:100]:  # Show first 100
                row_data = {
                    "Index": result['article_index'],
                    "Error": "Yes" if result['error'] else "No"
                }
                
                for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
                    gt = result['ground_truth'].get(factor, '')
                    pred = result['llm_output'].get(factor, '')
                    match = result['comparison_results'].get(factor, False)
                    
                    row_data[f"{factor}_GT"] = gt
                    row_data[f"{factor}_Pred"] = pred
                    row_data[f"{factor}_Match"] = "✅" if match else "❌"
                
                detailed_data.append(row_data)
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                st.dataframe(detailed_df, use_container_width=True)
            
            # LLM Responses Section
            st.write("### LLM Raw Responses")
            st.write("Click on each article to view the LLM's raw response, ground truth, and parsed output.")
            
            # Create expandable sections for each article's LLM response
            for idx, result in enumerate(results['results'][:50]):  # Show first 50
                headline_preview = result.get('headline', f"Article {result['article_index']}")
                if len(headline_preview) > 80:
                    headline_preview = headline_preview[:80] + "..."
                
                error_indicator = " ⚠️" if result.get('error') else ""
                with st.expander(f"Article {result['article_index']}: {headline_preview}{error_indicator}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Ground Truth:**")
                        gt_data = []
                        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
                            gt_val = result['ground_truth'].get(factor, 'N/A')
                            if gt_val and str(gt_val) != 'N/A':
                                gt_data.append(f"**{factor}:** {gt_val}")
                        if gt_data:
                            st.write("\n".join(gt_data))
                        else:
                            st.write("_No ground truth data_")
                    
                    with col2:
                        st.write("**LLM Output (Parsed):**")
                        llm_data = []
                        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
                            llm_val = result['llm_output'].get(factor, 'N/A')
                            match = result['comparison_results'].get(factor, False)
                            match_icon = "✅" if match else "❌"
                            if llm_val and str(llm_val) != 'N/A':
                                llm_data.append(f"**{factor}:** {llm_val} {match_icon}")
                        if llm_data:
                            st.write("\n".join(llm_data))
                        else:
                            st.write("_No LLM output_")
                    
                    # Show raw response
                    st.write("---")
                    st.write("**Raw LLM Response:**")
                    if result.get('raw_response'):
                        try:
                            # Try to format as JSON if possible
                            import json
                            parsed = json.loads(result['raw_response'])
                            st.json(parsed)
                        except:
                            # If not JSON, show as code
                            st.code(result['raw_response'], language='text')
                    elif result.get('error_message'):
                        st.error(f"**Error:** {result['error_message']}")
                    else:
                        st.info("No raw response available")
                    
                    if result.get('execution_time_ms'):
                        st.caption(f"Execution time: {result['execution_time_ms']:.2f} ms")
            
            # Export Results
            st.write("### Export Results")
            results_df = pd.DataFrame(results['results'])
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"results_{results['run_id']}.csv",
                mime="text/csv"
            )

elif page == "View Logs":
    st.header("View Evaluation Logs")
    
    master_log = logger.load_master_log()
    
    if master_log is not None and len(master_log) > 0:
        st.write(f"**Total logged evaluations:** {len(master_log)}")
        
        # Filters
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            run_ids = master_log['run_id'].unique().tolist()
            selected_run = st.selectbox(
                "Filter by Run ID",
                ["All"] + run_ids
            )
        
        with col2:
            models = master_log['model_name'].unique().tolist()
            selected_model = st.selectbox(
                "Filter by Model",
                ["All"] + models
            )
        
        # Apply filters
        filtered_log = master_log.copy()
        if selected_run != "All":
            filtered_log = filtered_log[filtered_log['run_id'] == selected_run]
        if selected_model != "All":
            filtered_log = filtered_log[filtered_log['model_name'] == selected_model]
        
        st.write(f"**Showing {len(filtered_log)} entries**")
        
        # Display log
        st.dataframe(filtered_log, use_container_width=True)
        
        # Download
        if st.button("📥 Download Filtered Log"):
            csv = filtered_log.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_log.csv",
                mime="text/csv"
            )
    else:
        st.info("No logs available yet. Run an evaluation first.")

elif page == "Past Runs":
    st.header("Past Evaluation Runs")
    
    runs = logger.list_runs()
    
    if runs:
        st.write(f"**Total runs:** {len(runs)}")
        
        for run in runs:
            with st.expander(f"Run: {run['run_id']}"):
                metadata = run.get('metadata', {})
                if metadata:
                    st.json(metadata)
                
                # Load run log
                run_log = logger.load_run_log(run['run_id'])
                if run_log is not None:
                    st.write(f"**Articles evaluated:** {len(run_log)}")
                    
                    # Quick stats
                    if len(run_log) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            errors = len(run_log[run_log['error_message'] != ''])
                            st.metric("Errors", errors)
                        with col2:
                            avg_time = run_log['execution_time_ms'].mean()
                            st.metric("Avg Time (ms)", f"{avg_time:.1f}")
                        with col3:
                            matches = run_log.filter(regex='^match_').sum().sum()
                            total = run_log.filter(regex='^match_').notna().sum().sum()
                            accuracy = matches / total if total > 0 else 0
                            st.metric("Overall Accuracy", f"{accuracy*100:.2f}%")
                    
                    if st.button(f"View Full Log", key=f"view_{run['run_id']}"):
                        st.dataframe(run_log, use_container_width=True)
    else:
        st.info("No past runs found.")
