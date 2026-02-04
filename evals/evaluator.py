"""
Core evaluation logic with parallel execution and comparison functions.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

from generative_models.LLM import generate
from evals.utils import (
    parse_llm_response,
    compare_numeric_values,
    compare_categorical_values,
    normalize_factor_name,
    NUMERIC_FACTORS,
    CATEGORICAL_FACTORS,
    get_body_column
)
from evals.logger import EvaluationLogger


class Evaluator:
    """Evaluator for running LLM evaluations on ground truth data."""
    
    def __init__(
        self,
        logger: EvaluationLogger,
        tolerance: float = 0.1,
        max_workers: Optional[int] = None
    ):
        self.logger = logger
        self.tolerance = tolerance
        self.max_workers = max_workers or min(10, os.cpu_count() or 1)
    
    def evaluate_article(
        self,
        article_index: int,
        row: pd.Series,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        parameters: Dict[str, Any],
        body_column: str,
        provider: str = "gemini"
    ) -> Dict[str, Any]:
        """
        Evaluate a single article.
        
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        error_message = None
        llm_output = {}
        raw_response = None
        
        try:
            # Prepare article dict
            article = {
                "title": str(row.get("headline", "")),
                "source": str(row.get("source", "Unknown")),
                "author": str(row.get("author", "Unknown")),
                "publication_date": str(row.get("publication_date", "Unknown")),
                "content": str(row.get(body_column, ""))
            }
            
            # Call LLM
            raw_response = generate(
                api_key=api_key,
                system_prompt=system_prompt,
                prompt=user_prompt,
                article=article,
                model=model,
                provider=provider,
                **parameters
            )
            
            # Parse response
            llm_output = parse_llm_response(raw_response)
            
        except Exception as e:
            error_message = str(e)
            llm_output = {}
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Extract ground truth
        ground_truth = {}
        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
            # Try different column name variations
            col_name = None
            for col in row.index:
                if normalize_factor_name(col) == factor:
                    col_name = col
                    break
            
            if col_name and pd.notna(row[col_name]):
                ground_truth[factor] = row[col_name]
        
        # Compare results
        comparison_results = {}
        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
            gt_value = ground_truth.get(factor)
            pred_value = llm_output.get(factor)
            
            if gt_value is None or pd.isna(gt_value):
                comparison_results[factor] = None
                continue
            
            if factor in NUMERIC_FACTORS:
                is_match, _ = compare_numeric_values(
                    pred_value,
                    gt_value,
                    tolerance=self.tolerance
                )
                comparison_results[factor] = is_match
            else:
                is_match = compare_categorical_values(
                    pred_value,
                    gt_value,
                    factor
                )
                comparison_results[factor] = is_match
        
        # Log result
        self.logger.log_result(
            article_index=article_index,
            headline=str(row.get("headline", "")),
            body=str(row.get(body_column, "")),
            url=row.get("url") if "url" in row.index else None,
            ground_truth=ground_truth,
            llm_output=llm_output,
            comparison_results=comparison_results,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            raw_response=raw_response,
            user_prompt=user_prompt,
            parameters=parameters
        )
        
        return {
            "article_index": article_index,
            "headline": str(row.get("headline", "")),
            "ground_truth": ground_truth,
            "llm_output": llm_output,
            "comparison_results": comparison_results,
            "error": error_message is not None,
            "error_message": error_message,
            "execution_time_ms": execution_time_ms,
            "raw_response": raw_response
        }
    
    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        parameters: Dict[str, Any],
        provider: str = "gemini",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset in parallel.
        
        Returns:
            Dictionary with evaluation results and metrics
        """
        body_column = get_body_column(df)
        if not body_column:
            raise ValueError("CSV must have 'body' or 'content' column")
        
        # Set logger context
        self.logger.set_run_context(model, system_prompt, user_prompt)
        
        # Start run
        run_id = self.logger.start_run(
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parameters=parameters,
            num_articles=len(df)
        )
        
        results = []
        errors = []
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.evaluate_article,
                    idx,
                    row,
                    api_key,
                    model,
                    system_prompt,
                    user_prompt,
                    parameters,
                    body_column,
                    provider
                ): idx
                for idx, row in df.iterrows()
            }
            
            # Process results as they complete
            # Use tqdm only if no progress callback (for CLI usage)
            if progress_callback:
                pbar = None
            else:
                pbar = tqdm(total=len(df), desc="Evaluating articles")
            
            try:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result["error"]:
                            errors.append(result)
                    except Exception as e:
                        errors.append({
                            "article_index": idx,
                            "error": True,
                            "error_message": str(e)
                        })
                    
                    if pbar:
                        pbar.update(1)
                    if progress_callback:
                        # Ensure progress doesn't exceed 1.0
                        completed = min(len(results) + len(errors), len(df))
                        progress_callback(completed, len(df))
            finally:
                if pbar:
                    pbar.close()
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, df)
        
        # End run
        summary = {
            "run_id": run_id,
            "total_articles": len(df),
            "successful": len(results) - len(errors),
            "errors": len(errors),
            "metrics": metrics
        }
        self.logger.end_run(summary)
        
        return {
            "run_id": run_id,
            "results": results,
            "errors": errors,
            "metrics": metrics,
            "summary": summary
        }
    
    def _calculate_metrics(
        self,
        results: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        from evals.utils import calculate_metrics
        
        all_metrics = {}
        
        # Collect data per factor
        for factor in NUMERIC_FACTORS + CATEGORICAL_FACTORS:
            ground_truth_list = []
            predictions_list = []
            
            for result in results:
                if result["error"]:
                    continue
                
                gt_value = result["ground_truth"].get(factor)
                pred_value = result["llm_output"].get(factor)
                
                if gt_value is not None and not pd.isna(gt_value):
                    ground_truth_list.append(gt_value)
                    predictions_list.append(pred_value if pred_value is not None else "")
            
            if ground_truth_list:
                is_numeric = factor in NUMERIC_FACTORS
                factor_metrics = calculate_metrics(
                    ground_truth_list,
                    predictions_list,
                    factor,
                    is_numeric=is_numeric,
                    tolerance=self.tolerance
                )
                all_metrics[factor] = factor_metrics
            else:
                all_metrics[factor] = {
                    "accuracy": 0.0,
                    "num_correct": 0,
                    "num_total": 0
                }
        
        # Calculate overall accuracy
        total_correct = 0
        total_comparisons = 0
        
        for result in results:
            if result["error"]:
                continue
            
            for factor, is_match in result["comparison_results"].items():
                if is_match is not None:
                    total_comparisons += 1
                    if is_match:
                        total_correct += 1
        
        overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0
        
        all_metrics["overall"] = {
            "accuracy": overall_accuracy,
            "num_correct": total_correct,
            "num_total": total_comparisons
        }
        
        return all_metrics
