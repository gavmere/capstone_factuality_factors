"""
Logging functionality for tracking all evaluation runs.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import hashlib


class EvaluationLogger:
    """Logger for evaluation runs."""

    def __init__(self, logs_dir: str = "evals/logs"):
        self.logs_dir = logs_dir
        self.metadata_dir = os.path.join(logs_dir, "metadata")

        # Create directories if they don't exist
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.master_log_path = os.path.join(logs_dir, "master_log.csv")
        self.current_run_id = None
        self.current_log_path = None

    def start_run(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        parameters: Dict[str, Any],
        num_articles: int,
    ) -> str:
        """
        Start a new evaluation run.

        Returns:
            Run ID (timestamp string)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_id = timestamp

        # Create run-specific log file
        self.current_log_path = os.path.join(
            self.logs_dir, f"evaluation_logs_{timestamp}.csv"
        )

        # Save metadata (full prompts)
        metadata = {
            "run_id": timestamp,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "parameters": parameters,
            "num_articles": num_articles,
            "start_time": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.metadata_dir, f"metadata_{timestamp}.json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Initialize log file with headers
        self._write_header()

        return timestamp

    def _write_header(self):
        """Write header to log file."""
        headers = [
            "run_id",
            "timestamp",
            "article_index",
            "headline",
            "body_preview",
            "url",
            "model_name",
            "system_prompt_hash",
            "user_prompt",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            # Ground truth columns
            "gt_clickbait",
            "gt_headline_body_relation",
            "gt_political_affiliation",
            "gt_sensationalism",
            "gt_sentiment_analysis",
            "gt_toxicity",
            # LLM output columns
            "llm_clickbait",
            "llm_headline_body_relation",
            "llm_political_affiliation",
            "llm_sensationalism",
            "llm_sentiment_analysis",
            "llm_toxicity",
            # Comparison results
            "match_clickbait",
            "match_headline_body_relation",
            "match_political_affiliation",
            "match_sensationalism",
            "match_sentiment_analysis",
            "match_toxicity",
            # Error information
            "error_message",
            "execution_time_ms",
            "raw_llm_response",
        ]

        with open(self.current_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_result(
        self,
        article_index: int,
        headline: str,
        body: str,
        url: Optional[str],
        ground_truth: Dict[str, Any],
        llm_output: Dict[str, Any],
        comparison_results: Dict[str, bool],
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        raw_response: Optional[str] = None,
        system_prompt_hash: Optional[str] = None,
        user_prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Log a single evaluation result."""
        if not self.current_log_path:
            raise ValueError("No active run. Call start_run() first.")

        # Truncate body for preview (first 200 chars)
        body_preview = body[:200] + "..." if len(body) > 200 else body

        # Hash system prompt if not provided
        if system_prompt_hash is None and hasattr(self, "_current_system_prompt"):
            system_prompt_hash = self._hash_prompt(self._current_system_prompt)

        row = {
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "article_index": article_index,
            "headline": headline,
            "body_preview": body_preview,
            "url": url or "",
            "model_name": getattr(self, "_current_model_name", ""),
            "system_prompt_hash": system_prompt_hash or "",
            "user_prompt": user_prompt or getattr(self, "_current_user_prompt", ""),
            "temperature": parameters.get("temperature", "") if parameters else "",
            "max_tokens": parameters.get("max_tokens", "") if parameters else "",
            "top_p": parameters.get("top_p", "") if parameters else "",
            "top_k": parameters.get("top_k", "") if parameters else "",
            # Ground truth
            "gt_clickbait": ground_truth.get("Clickbait", ""),
            "gt_headline_body_relation": ground_truth.get("Headline-Body-Relation", ""),
            "gt_political_affiliation": ground_truth.get("Political Affiliation", ""),
            "gt_sensationalism": ground_truth.get("Sensationalism", ""),
            "gt_sentiment_analysis": ground_truth.get("Sentiment Analysis", ""),
            "gt_toxicity": ground_truth.get("Toxicity", ""),
            # LLM output
            "llm_clickbait": llm_output.get("Clickbait", ""),
            "llm_headline_body_relation": llm_output.get("Headline-Body-Relation", ""),
            "llm_political_affiliation": llm_output.get("Political Affiliation", ""),
            "llm_sensationalism": llm_output.get("Sensationalism", ""),
            "llm_sentiment_analysis": llm_output.get("Sentiment Analysis", ""),
            "llm_toxicity": llm_output.get("Toxicity", ""),
            # Comparison
            "match_clickbait": comparison_results.get("Clickbait", False),
            "match_headline_body_relation": comparison_results.get(
                "Headline-Body-Relation", False
            ),
            "match_political_affiliation": comparison_results.get(
                "Political Affiliation", False
            ),
            "match_sensationalism": comparison_results.get("Sensationalism", False),
            "match_sentiment_analysis": comparison_results.get(
                "Sentiment Analysis", False
            ),
            "match_toxicity": comparison_results.get("Toxicity", False),
            # Error info
            "error_message": error_message or "",
            "execution_time_ms": execution_time_ms or "",
            "raw_llm_response": raw_response or "",
        }

        # Append to run log
        with open(self.current_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        # Append to master log
        if os.path.exists(self.master_log_path):
            with open(self.master_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
        else:
            # Create master log with header
            with open(self.master_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

    def set_run_context(self, model_name: str, system_prompt: str, user_prompt: str):
        """Set context for current run (for logging)."""
        self._current_model_name = model_name
        self._current_system_prompt = system_prompt
        self._current_user_prompt = user_prompt

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for storage."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]

    def end_run(self, summary: Optional[Dict[str, Any]] = None):
        """End the current run and optionally save summary."""
        if self.current_run_id and summary:
            summary_path = os.path.join(
                self.metadata_dir, f"summary_{self.current_run_id}.json"
            )
            summary["end_time"] = datetime.now().isoformat()
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        self.current_run_id = None
        self.current_log_path = None

    def load_master_log(self) -> Optional[pd.DataFrame]:
        """Load the master log as a DataFrame."""
        if os.path.exists(self.master_log_path):
            try:
                return pd.read_csv(self.master_log_path)
            except Exception:
                return None
        return None

    def load_run_log(self, run_id: str) -> Optional[pd.DataFrame]:
        """Load a specific run log."""
        log_path = os.path.join(self.logs_dir, f"evaluation_logs_{run_id}.csv")
        if os.path.exists(log_path):
            try:
                return pd.read_csv(log_path)
            except Exception:
                return None
        return None

    def load_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a specific run."""
        metadata_path = os.path.join(self.metadata_dir, f"metadata_{run_id}.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all evaluation runs."""
        runs = []

        # Get all log files
        if os.path.exists(self.logs_dir):
            for filename in os.listdir(self.logs_dir):
                if filename.startswith("evaluation_logs_") and filename.endswith(
                    ".csv"
                ):
                    run_id = filename.replace("evaluation_logs_", "").replace(
                        ".csv", ""
                    )
                    metadata = self.load_run_metadata(run_id)

                    run_info = {
                        "run_id": run_id,
                        "log_file": filename,
                        "metadata": metadata,
                    }
                    runs.append(run_info)

        # Sort by run_id (timestamp) descending
        runs.sort(key=lambda x: x["run_id"], reverse=True)
        return runs
