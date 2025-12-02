"""
DSBC (Data Science Benchmarking and Comparison) Evaluation System

A professional evaluation framework for comparing Large Language Models on data science tasks.
Supports both full dataset evaluation and single query testing with LLM-as-judge evaluation.

Usage:
    # Full dataset evaluation
    python main.py

    # Single query test
    python main.py --mode single --query "What is the average sales?" --filepath temp/datafiles/sales_dataset.csv

Author: DSBC Team
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

from evaluations.llm_as_judge import run_evals
from generate_answers.dsbc_answers import run_dataset, run_single_query

warnings.filterwarnings('ignore', category=UserWarning)


class DSBCEvaluator:
    """Professional DSBC evaluation system for LLM benchmarking."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.3,
        judge_provider: str = "vertex_ai",
        judge_model: str = "gemini-2.0-flash",
        judge_temperature: float = 0.2
    ):
        """
        Initialize DSBC evaluator.

        Args:
            provider: LLM provider for generation (required)
            model_name: Model name for generation (required)
            temperature: Generation temperature (default: 0.3)
            judge_provider: LLM provider for evaluation (default: vertex_ai)
            judge_model: Model name for evaluation (default: gemini-2.0-flash)
            judge_temperature: Judge temperature (default: 0.2)
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature

    def run_dataset_evaluation(self, no_of_samples: int | None = None) -> str:
        """
        Run complete dataset evaluation with LLM judge.

        Returns:
            Filename of processed results
        """
       
        print("DSBC DATASET EVALUATION") 
        print(f"Provider: {self.provider}")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"Judge Provider: {self.judge_provider}")
        print(f"Judge Model: {self.judge_model}")
        if no_of_samples is not None:
            print(f"Sample size: {no_of_samples} rows")
        print("-" * 70)

        try:
            print("\nGenerating responses for all datasets...")
            file_name = run_dataset(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                no_of_samples=no_of_samples,
            )

            print("Evaluating responses using LLM as judge...")
            processed_data_path = f"temp/processed_data/{file_name}"

            run_evals(
                df_name=file_name,
                input_path=processed_data_path,
                provider=self.judge_provider,
                model_name=self.judge_model
            )

            print("\nEVALUATION COMPLETE")
            print(f"Results saved in: temp/evaluated_data/")
            print(f"Processed data: {processed_data_path}")
            

            return file_name

        except Exception as e:
            print(f"Error during dataset evaluation: {e}")
            raise

    def run_single_query_evaluation(self, query: str, filepath: str) -> Dict[str, Any]:
        """
        Run single query evaluation on specified dataset.

        Args:
            query: Data science question to answer
            filepath: Path to CSV dataset file

        Returns:
            Dictionary containing the evaluation results
        """
       
        print("DSBC SINGLE QUERY EVALUATION")
        print(f"Query: {query}")
        print(f"Dataset: {filepath}")
        print(f"Provider: {self.provider}")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print("-" * 70)

        try:
            run_single_query(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                query=query,
                filepath=filepath
            )

         
            print(f"Status: Success")
            print(f"Query processed successfully on {filepath}")
         

            return {"status": "success", "filepath": filepath, "query": query}

        except Exception as e:
            print(f" Error during single query evaluation: {e}")
            raise


def create_evaluator_from_args(args) -> DSBCEvaluator:
    """Create evaluator instance from command line arguments."""
    return DSBCEvaluator(
        provider=args.provider,
        model_name=args.model,
        temperature=getattr(args, 'temperature', 0.3),
        judge_provider=getattr(args, 'judge_provider', 'vertex_ai'),
        judge_model=getattr(args, 'judge_model', 'gemini-2.0-flash'),
        judge_temperature=getattr(args, 'judge_temperature', 0.2)
    )


def main():
    """Main entry point with professional command-line interface."""
    parser = argparse.ArgumentParser(
        prog="dsbc-eval",
        description="DSBC Evaluation System - Professional LLM Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset evaluation (requires 4 parameters: generation + judge)
  python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider vertex_ai --judge-model gemini-2.0-flash

  # Single query (requires only 2 parameters + query/filepath)
  python main.py --mode single --query "What is the average sales?" --filepath temp/datafiles/sales_dataset.csv --provider claude --model claude-3.5-sonnet-20241022

  # Custom judge temperature
  python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider openai --judge-model gpt-4 --judge-temperature 0.1

Supported Providers: vertex_ai, openai, claude, gemini, azure, openrouter, anthropic_vertex
        """
    )

    # Common arguments
    parser.add_argument('--provider',
                       help='LLM provider for generation (required for full dataset mode and single mode)')
    parser.add_argument('--model',
                       help='Model name for generation (required for full dataset mode and single mode)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for generation (default: 0.3)')

    # Judge arguments (only for dataset mode)
    parser.add_argument('--judge-provider',
                       help='LLM provider for evaluation (required for dataset mode)')
    parser.add_argument('--judge-model',
                       help='Model name for evaluation (required for dataset mode)')
    parser.add_argument('--judge-temperature', type=float, default=0.2,
                       help='Temperature for evaluation (default: 0.2)')

    # Mode selection
    parser.add_argument('--mode', choices=['dataset', 'single'], default='dataset',
                       help='Evaluation mode: dataset (default) or single query')

    # Dataset sampling argument (only for dataset mode)
    parser.add_argument('--sample-size', type=int,
                       help='Optional number of rows to sample from the dataset for faster runs')

    # Evaluation-only on existing processed file (skip generation)
    parser.add_argument('--processed-file',
                       help='Path to an existing processed CSV (with model answers) to run evaluation only')

    # Single query arguments
    parser.add_argument('--query', help='Data science question for single query mode (required for single mode)')
    parser.add_argument('--filepath', help='Path to CSV dataset file for single query mode (required for single mode)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'dataset':
        # Judge is always required for dataset mode
        if not args.judge_provider:
            parser.error("--judge-provider is required for dataset mode")
        if not args.judge_model:
            parser.error("--judge-model is required for dataset mode")

        # If user provided a processed file, ensure it exists
        if args.processed_file and not Path(args.processed_file).exists():
            parser.error(f"Processed file not found: {args.processed_file}")

        # For full pipeline (no processed-file), require generation provider/model
        if not args.processed_file:
            if not args.provider:
                parser.error("--provider is required for dataset mode when not using --processed-file")
            if not args.model:
                parser.error("--model is required for dataset mode when not using --processed-file")
    else:  # single mode
        # Single mode requires generation parameters + query/filepath
        if not args.provider:
            parser.error("--provider is required for single mode")
        if not args.model:
            parser.error("--model is required for single mode")
        if not args.query:
            parser.error("--query is required for single mode")
        if not args.filepath:
            parser.error("--filepath is required for single mode")
        if not Path(args.filepath).exists():
            parser.error(f"Dataset file not found: {args.filepath}")

    # Create evaluator and run evaluation
    try:
        # Dataset mode
        if args.mode == 'dataset' and args.processed_file:
            # EVAL-ONLY: use only judge provider/model, skip generation provider/model entirely
            processed_path = Path(args.processed_file)
            file_name = processed_path.name

            print("DSBC DATASET EVALUATION (EVAL-ONLY MODE)")
            print(f"Judge Provider: {args.judge_provider}")
            print(f"Judge Model: {args.judge_model}")
            print(f"Processed file: {processed_path}")
            print("-" * 70)

            run_evals(
                df_name=file_name,
                input_path=str(processed_path),
                provider=args.judge_provider,
                model_name=args.judge_model
            )
        else:
            # All other modes use the evaluator (requires provider/model)
            evaluator = create_evaluator_from_args(args)

            if args.mode == 'dataset':
                # Full pipeline: generation + evaluation
                evaluator.run_dataset_evaluation(
                    no_of_samples=getattr(args, 'sample_size', None)
                )
            else:  # single mode
                evaluator.run_single_query_evaluation(args.query, args.filepath)

    except KeyboardInterrupt:
        print("\n  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()