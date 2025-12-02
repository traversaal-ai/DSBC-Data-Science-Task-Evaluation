"""
SEPARATE UTILITY FUNCTION TO GET METRICS FROM DSBC EVALUATION
"""

from pathlib import Path
from typing import Dict, Any, Union
from evaluations.llm_as_judge import run_evals


def get_evaluation_metrics(
    processed_file: str,
    judge_provider: str,
    judge_model: str
) -> Union[Dict[str, Any], tuple, Any]:
    """
    Run evaluation on a processed file and return metrics.
    
    This is a standalone function that doesn't interfere with CLI commands.
    
    Args:
        processed_file: Path to processed CSV file with model answers
        judge_provider: LLM provider for evaluation (e.g., 'azure', 'vertex_ai')
        judge_model: Model name for evaluation (e.g., 'gpt-4o-data-science')
    
    Returns:
        Evaluation metrics (format depends on run_evals implementation)
        Could be dict, tuple, or other type
    
    Example:
        metrics = get_evaluation_metrics(
            processed_file='/path/to/file.csv',
            judge_provider='azure',
            judge_model='gpt-4o-data-science'
        )
    """
    # Validate file exists
    processed_path = Path(processed_file)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    
    # Extract filename
    file_name = processed_path.name
    
    print("="*70)
    print("GETTING EVALUATION METRICS (PROGRAMMATIC MODE)")
    print("="*70)
    print(f"Judge Provider: {judge_provider}")
    print(f"Judge Model: {judge_model}")
    print(f"Processed file: {processed_path}")
    print("-" * 70)
    
    # Run evaluation and capture results
    try:
        eval_results = run_evals(
            df_name=file_name,
            input_path=str(processed_path),
            provider=judge_provider,
            model_name=judge_model
        )
        
        print("\n✓ Evaluation complete!")
        print(f"Results type: {type(eval_results).__name__}")
        
        # Print results based on type
        _print_results(eval_results)
        
        return eval_results
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        raise


def _print_results(results):
    """Helper to print results in a readable format."""
    print("\nRESULTS:")
    print("-" * 70)
    
    if results is None:
        print("No results returned")
    
    elif isinstance(results, dict):
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    elif isinstance(results, tuple):
        for i, value in enumerate(results, 1):
            print(f"  Metric {i}: {value}")
    
    elif isinstance(results, (int, float, str)):
        print(f"  Value: {results}")
    
    else:
        print(f"  {results}")
    
    print("="*70)


# Convenience function for batch evaluation
def batch_evaluate_files(
    processed_files: list,
    judge_provider: str,
    judge_model: str
) -> Dict[str, Any]:
    """
    Evaluate multiple processed files and return all metrics.
    
    Args:
        processed_files: List of paths to processed CSV files
        judge_provider: LLM provider for evaluation
        judge_model: Model name for evaluation
    
    Returns:
        Dictionary mapping filenames to their metrics
    
    Example:
        files = [
            '/path/to/model1.csv',
            '/path/to/model2.csv',
        ]
        
        all_metrics = batch_evaluate_files(
            processed_files=files,
            judge_provider='azure',
            judge_model='gpt-4o-data-science'
        )
        
        for file, metrics in all_metrics.items():
            print(f"{file}: {metrics}")
    """
    results = {}
    
    print("\n" + "="*70)
    print(f"BATCH EVALUATION: {len(processed_files)} files")
    print("="*70 + "\n")
    
    for i, file_path in enumerate(processed_files, 1):
        print(f"\n[{i}/{len(processed_files)}] Processing: {Path(file_path).name}")
        
        try:
            metrics = get_evaluation_metrics(
                processed_file=file_path,
                judge_provider=judge_provider,
                judge_model=judge_model
            )
            results[Path(file_path).name] = metrics
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[Path(file_path).name] = None
    
    print("\n" + "="*70)
    print("BATCH EVALUATION COMPLETE")
    print("="*70)
    
    return results