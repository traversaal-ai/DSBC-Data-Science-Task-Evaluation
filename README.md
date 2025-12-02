# Data Science Code Generation Evaluation

This repository evaluates Large Language Models on the [DSBC (Data Science Benchmarking)](https://huggingface.co/datasets/large-traversaal/dsbc_v0) dataset. It systematically tests LLM capabilities in data science code generation by generating responses to complex data science questions and evaluating them using LLM-based judges.






## Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys to .env file
# (see Configuration section below)

# 3. Run evaluation
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider vertex_ai --judge-model gemini-2.0-flash
python main.py --mode single --query "Your question" --filepath data.csv --provider claude --model claude-3.5-sonnet-20241022

# Optional: Download fresh datasets if CSV files cause issues
python -c "from generate_answers.datascience.load_datafiles import download_datasets; download_datasets()"
```

## Features

- **Multi-Provider LLM Support**: OpenAI, Google Gemini, Anthropic Claude, Vertex AI, Azure OpenAI, OpenRouter, and more
- **Two Modes**: Full dataset evaluation or single query testing
- **LLM-as-Judge Evaluation**: Uses LLMs to score and evaluate responses
- **CLI**: Clean, argparse-based command-line interface

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Copy `env.example` to `.env` and configure your API keys:
    ```bash
    cp env.example .env
    ```

3. **Python**:

```python
from main import DSBCEvaluator

# Create evaluator
evaluator = DSBCEvaluator(
    provider="vertex_ai",
    model_name="gemini-2.5-pro",
    temperature=0.3,
    judge_provider="openai",
    judge_model="gpt-4"
)

# Run dataset evaluation
result_filename = evaluator.run_dataset_evaluation()

# Run single query
result = evaluator.run_single_query_evaluation(
    query="What is the average sales?",
    filepath="temp/datafiles/sales_dataset.csv"
)
```


4. **Run evaluation CLI**:
   ```bash
   # Full dataset evaluation (default)
   python main.py

   # Single query test
   python main.py --mode single --query "What is the average sales?" --filepath temp/datafiles/sales_dataset.csv
   ```

## Running the Dataset Evaluations

### Complete Setup Guide

#### Step 1: Install Dependencies
```bash
cd path/to/dsbc_evaluation
pip install -r requirements.txt
```

#### Step 2: Configure API Keys
Copy `env.example` to `.env` and configure your API keys:
    ```bash
    cp env.example .env
    ```

```bash
# Basic usage (all 4 parameters required)
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider vertex_ai --judge-model gemini-2.0-flash

# Custom generation model
python main.py --provider openai --model gpt-4 --judge-provider vertex_ai --judge-model gemini-2.0-flash --temperature 0.7

# Custom judge model
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider claude --judge-model claude-3-5-sonnet-20241022
```

### Single Query Testing Mode
Test specific questions on specific datasets (requires only 2 parameters + query/filepath):

```bash
# Basic single query (only generation parameters needed)
python main.py --mode single --query "What is the average sales by region?" --filepath temp/datafiles/sales_dataset.csv --provider vertex_ai --model gemini-2.5-pro

# Different provider and model
python main.py --mode single --query "Calculate correlation between price and sales" --filepath temp/datafiles/sales_dataset.csv --provider claude --model claude-3.5-sonnet-20241022



# Evaluate with GPT-4 as judge instead of Gemini
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider openai --judge-model gpt-4

```


## Usage

### Command Line Interface

```bash
dsbc-eval [OPTIONS]

Options:
  --provider TEXT           LLM provider for generation (required)
  --model TEXT              Model name for generation (required)
  --temperature FLOAT       Generation temperature (default: 0.3)
  --judge-provider TEXT     LLM provider for evaluation (required for dataset mode)
  --judge-model TEXT        Judge model name for evaluation (required for dataset mode)
  --judge-temperature FLOAT Judge temperature (default: 0.2)
  --mode [dataset|single]   Evaluation mode (default: dataset)
  --sample-size INTEGER     Optional number of rows to sample from the benchmark
  --query TEXT              Query for single mode (required for single mode)
  --filepath TEXT           Dataset file path for single mode (required for single mode)
  --help                    Show this message and exit
```

### Examples

```bash
# Full benchmark
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider vertex_ai --judge-model gemini-2.0-flash

# Faster run on a sample of 50 rows
python main.py --provider vertex_ai --model gemini-2.5-pro --judge-provider vertex_ai --judge-model gemini-2.0-flash --sample-size 50

# Single query
python main.py --mode single --query "What is the average sales?" --filepath temp/datafiles/sales_dataset.csv --provider claude --model claude-3.5-sonnet-20241022
```



## Supported LLM Providers

| Provider | Models | Setup Required |
|----------|--------|----------------|
| **vertex_ai** | Gemini 2.5 Pro,| GCP credentials |
| **anthropic_vertex** | Claude Sonnet 4 | GCP credentials |
| **openai** |GPT-5 | API key |
| **claude** | Claude Sonnet 4, | API key |
| **gemini** | Gemini 2.5 Pro | API key |
| **azure** | GPT-4 on Azure | Azure credentials |
| **openrouter** | Various models (gpt-5.1-codex, gpt-o4-mini, glm-4.5) | API key |

See `generate_answers/ReadMe` for detailed provider setup instructions.

## Project Structure

```
dsbc_evaluation/
├── main.py                    # Main CLI and DSBCEvaluator class
├── requirements.txt           # Python dependencies
├── generate_answers/          # Answer generation module
│   ├── llm_clients.py        # Multi-provider LLM client library
│   ├── dsbc_answers.py       # Dataset processing logic
│   └── datascience/          # Data science utilities
├── evaluations/               # Evaluation module
│   └── llm_as_judge.py       # LLM-based evaluation
└── temp/                     # Data and results directory
    ├── datafiles/            # Input datasets (CSV files)
    │   └── *Note: If CSV files cause issues, run download_datasets() from generate_answers/datascience/load_datafiles.py to get fresh datasets from source
    ├── processed_data/       # Generated responses
    ├── evaluated_data/       # Evaluation results
    └── results/              # Final evaluation summaries
```





## Output Files

- **Processed Data**: `temp/processed_data/{model_name}_{uuid}.csv`
- **Evaluated Data**: `temp/evaluated_data/evaluated_{model1}_{model2}_{uuid}.csv`
- **Results**: `temp/results/evaluation_{filename}.json`

## Evaluation Results

### Answer Generation Settings
- **Temperature**: 0.3 (used for all model generations)

### Model Performance Scores
The following scores were obtained using LLM-as-Judge evaluation methodology:

| Model | Score |
|-------|--------|
| olive | 0.903 |
| claude-sonnet-4 | 0.751 |
| gemini-2.5-pro | 0.608 |
| gpt-5.1-codex | 0.728 |
| gpt-o4-mini | 0.618 |
| glm-4.5 | 0.673 |  

<img width="606" height="331" alt="image" src="https://github.com/user-attachments/assets/bb161c26-d516-4830-be58-ab98ab57b096" />

### Evaluation Settings
- **Judge Model**: gemini-flash-2.0 from Vertex AI
- **Judge Temperature**: 0.2 (default)

## Contributing

1. Add new LLM providers to `llm_clients.py`
2. Update provider documentation in `generate_answers/ReadMe`
3. Add new evaluation metrics to `evaluations/llm_as_judge.py`
4. Update tests and documentation


