
import os
import json
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm
import re
from generate_answers.llm_clients import LLMPipeline, LLMConfig, LLMProvider, load_config_from_env
from generate_answers.datascience.llm_response import answer_question
from evaluations.prompt import prompt_start, prompt_end
from dotenv import load_dotenv

load_dotenv()

def parse_tasks_from_string(tasks_str: str) -> List[str]:
    """Parse tasks from a string into a list of tasks"""
    if not tasks_str or pd.isna(tasks_str):
        return []
    # Split by comma and strip whitespace
    return [task.strip() for task in tasks_str.split(',')]

def calculate_evaluation_metrics( df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for LLM-as-a-judge results"""
    
    # Overall metrics
    yes_count = sum(1 for _, row in df.iterrows() if row.get("evaluation_result") == "Yes")
    no_count = sum(1 for _, row in df.iterrows() if row.get("evaluation_result") == "No")
    unknown_count = sum(1 for _, row in df.iterrows() if row.get("evaluation_result") == "Unknown")
    
    total_evaluated = yes_count + no_count
    total_entries = len(df)
    
    overall_accuracy = (yes_count / total_evaluated) if total_evaluated > 0 else 0
    success_rate = (total_evaluated / total_entries) if total_entries > 0 else 0
    
    # Task-specific metrics
    task_metrics = defaultdict(lambda: {"yes": 0, "no": 0, "unknown": 0, "total": 0})
    
    for _, row in df.iterrows():
        tasks = parse_tasks_from_string(row.get("Tasks", ""))
        evaluation = row.get("evaluation_result", "")
        
        # Count for each task this entry belongs to
        for task in tasks:
            task_metrics[task]["total"] += 1
            if evaluation == "Yes":
                task_metrics[task]["yes"] += 1
            elif evaluation == "No":
                task_metrics[task]["no"] += 1
            else:
                task_metrics[task]["unknown"] += 1
    
    # Calculate accuracy per task
    task_accuracies = {}
    for task, metrics in task_metrics.items():
        evaluated_count = metrics["yes"] + metrics["no"]
        if evaluated_count > 0:
            accuracy = metrics["yes"] / evaluated_count
            task_accuracies[task] = {
                "accuracy": accuracy,
                "yes_count": metrics["yes"],
                "no_count": metrics["no"],
                "unknown_count": metrics["unknown"],
                "total_count": metrics["total"],
                "evaluated_count": evaluated_count
            }
    
    # Dataset distribution
    dataset_metrics = defaultdict(lambda: {"yes": 0, "no": 0, "unknown": 0, "total": 0})
    for _, row in df.iterrows():
        dataset = row.get("Dataset", "Unknown")
        evaluation = row.get("evaluation_result", "")
        
        dataset_metrics[dataset]["total"] += 1
        if evaluation == "Yes":
            dataset_metrics[dataset]["yes"] += 1
        elif evaluation == "No":
            dataset_metrics[dataset]["no"] += 1
        else:
            dataset_metrics[dataset]["unknown"] += 1
    
    dataset_accuracies = {}
    for dataset, metrics in dataset_metrics.items():
        evaluated_count = metrics["yes"] + metrics["no"]
        if evaluated_count > 0:
            accuracy = metrics["yes"] / evaluated_count
            dataset_accuracies[dataset] = {
                "accuracy": accuracy,
                "yes_count": metrics["yes"],
                "no_count": metrics["no"],
                "unknown_count": metrics["unknown"],
                "total_count": metrics["total"],
                "evaluated_count": evaluated_count
            }
    
    return {
        "overall": {
            "accuracy": overall_accuracy,
            "success_rate": success_rate,
            "yes_count": yes_count,
            "no_count": no_count,
            "unknown_count": unknown_count,
            "total_entries": total_entries,
            "total_evaluated": total_evaluated
        },
        "tasks": task_accuracies,
        "datasets": dataset_accuracies
    }

def save_evaluation_results(df_name:str, metrics: Dict[str, Any], model_name: str) -> str:
    """Save evaluation metrics to a JSON file"""

    os.makedirs('temp/results', exist_ok=True)
    
    filepath = f'temp/results/evaluation_{df_name}.json'
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation results saved to: {filepath}")
    return filepath


def evaluate_llm_as_judge(df_name:str, pipeline: LLMPipeline, model_name: str, df: pd.DataFrame, detailed_report: bool) -> None:
    """Evaluate LLM responses using LLM-as-a-judge approach"""
    df['raw_judgment'] = '' 
    df['evaluation_result'] = ''  

    for i in tqdm(range(len(df)), desc="Processing rows"):
        Q = df['Query_Clean'][i]
        R = df['Response_output'][i]
        A = df['Response_Expected'][i]
        C = df['Solution_Code'][i]
        S = df['Response_code'][i]
        E = df['Response_reasoning'][i]
        prompt_end_formatted = prompt_end.format(Q=Q, A=A, C=C, R=R, S=S, E=E)
        temperature = 0.2
        final_template = prompt_start + "\n" + prompt_end_formatted
        judgment = answer_question(final_template, pipeline, model_name, temperature)
        df.at[i, 'raw_judgment'] = judgment
    
    df['evaluation_result'] = df['raw_judgment'].apply(
        lambda x: 'Yes' if re.search(r'\bYes\b', x) else ('No' if re.search(r'\bNo\b', x) else 'Unknown')
    )
    
    print("\n=== EVALUATION COUNTS ===")
    print(df['evaluation_result'].value_counts())
    
    metrics = calculate_evaluation_metrics(df)
    
    print("\n=== DETAILED EVALUATION METRICS ===")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"Success Rate: {metrics['overall']['success_rate']:.2%}")
    print(f"Yes Count: {metrics['overall']['yes_count']}")
    print(f"No Count: {metrics['overall']['no_count']}")
    print(f"Unknown Count: {metrics['overall']['unknown_count']}")
    print(f"Total Entries: {metrics['overall']['total_entries']}")
    print(f"Total Evaluated: {metrics['overall']['total_evaluated']}")
    
    print("\n=== TASK-SPECIFIC METRICS ===")
    for task, task_metrics in metrics['tasks'].items():
        print(f"{task}: {task_metrics['accuracy']:.2%} ({task_metrics['yes_count']}/{task_metrics['evaluated_count']})")
    
    print("\n=== DATASET-SPECIFIC METRICS ===")
    for dataset, dataset_metrics in metrics['datasets'].items():
        print(f"{dataset}: {dataset_metrics['accuracy']:.2%} ({dataset_metrics['yes_count']}/{dataset_metrics['evaluated_count']})")
    
    save_evaluation_results(df_name, metrics, model_name)
    
    return df

def run_evals(df_name:str, input_path: str, provider: str, model_name: str ):
    """Main function to run evaluation on a specific DataFrame file"""

    import sys; sys.stdout.flush(); print( "!!! Evaluations Started !!!\n", flush=True); sys.stdout.flush()

    config = load_config_from_env(provider)
    
    # Allow user to override model name
    if model_name:
        config.model_name = model_name
        MODEL_NAME = model_name
    else:
        MODEL_NAME = config.model_name
    
    print(f"Provider: {provider}")
    print(f"Model: {MODEL_NAME}")

    pipeline = LLMPipeline(config)
    
    
    print(f"Loading data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
  
    print("Starting evaluation...")
    evaluated_df = evaluate_llm_as_judge(df_name, pipeline, MODEL_NAME, df, detailed_report=True)
    
    os.makedirs('temp/evaluated_data', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    evaluated_path = f'temp/evaluated_data/evaluated_{MODEL_NAME.replace("/", "_")}_{base_name}.csv'
    evaluated_df.to_csv(evaluated_path, index=False)
    print(f"Evaluated data saved to: {evaluated_path}")
