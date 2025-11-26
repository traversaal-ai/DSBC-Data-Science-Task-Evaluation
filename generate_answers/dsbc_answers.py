import os
import uuid
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_answers.datascience.ds_structure import analyze_csv_stream
from generate_answers.datascience.load_datafiles import download_datasets
from generate_answers.llm_clients import LLMPipeline, LLMConfig, LLMProvider, load_config_from_env
from generate_answers.datascience.load_dataset import get_dataset_path_mapping

# Load environment variables
load_dotenv()

# Constants
NUM_WORKERS = 16     # ON CPU RECOMMENDED MAX IS 16 TO AVOID THREAD LOCKS




def get_single_response(
    query: str, 
    model_name: str, 
    temperature: float, 
    filepath: str, 
    llm_pipeline: LLMPipeline
) -> Dict[str, Any]:
    try:
        print("Answering your query ...")
        result = analyze_csv_stream(model_name, temperature, filepath, query, llm_pipeline)
        print("Query answered!")
        return {
            "status": "sucess",
            "code": result.get('Response_Code', ''),
            "output": result.get('Response_Output', ''),
            "analysis": result.get('Response_Analysis', ''),
            "reasoning": result.get('Response_Reasoning', '')
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            "status":"error",
            "code": "",
            "output": f"Error: {str(e)}",
            "analysis": "Failed to generate analysis due to error.",
            "reasoning": "Error occurred during data processing."
        }

def save_processed_data(df: pd.DataFrame, model_name: str) -> str:
    """
    Save the processed DataFrame with model name and UUID to temp/processed_data directory.
    
    Args:
        df: The processed DataFrame to save
        model_name: Name of the model used for processing
        
    Returns:
        Path to the saved CSV file
    """
    # Generate a 4-character UUID
    run_id = uuid.uuid4().hex[:4]
    
    # Clean model name for use in filename (remove slashes and other problematic characters)
    clean_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Create the directory if it doesn't exist
    os.makedirs('temp/processed_data', exist_ok=True)
    
    # Save the DataFrame to CSV
    file_name=f"{clean_model_name}_{run_id}.csv"
    filepath = f'temp/processed_data/{file_name}'
    df.to_csv(filepath, index=False)
    
    print(f"Processed data saved to: {filepath}")
    return file_name, filepath

def process_row(idx, row, model_name, temperature, pipeline):
    try:
        # Get query and path from the specific row
        query = row['Query_Clean']
        filepath = row['PATH']

        result = analyze_csv_stream(model_name, temperature, filepath, query, pipeline)

        # Check for errors
        if 'error' in result:
            return idx, None, None, result['error'], None

        code_2 = result['Response_Code']
        output_2 = result['Response_Output']
        analysis_2 = result['Response_Analysis']
        reasoning_2 = result['Response_Reasoning']

        return idx, code_2, output_2, analysis_2, reasoning_2
    except Exception as e:
        print(f"ERROR on row {idx}: {str(e)}")
        return idx, None, None, str(e), None

def get_dataset_responses( 
    download_data: bool,
    model_name: str, 
    temperature: float, 
    llm_pipeline: LLMPipeline
) -> Dict[str, Any]:
    if download_data:
        print("Downloading datasets...")
        download_datasets()
        print("Datasets downloaded!\nLoading dataset...")
    df_final = get_dataset_path_mapping()
    print("Dataset loaded!")
    
    # Initialize new columns if they don't exist
    for col in ['Response_code', 'Response_output', 'Response_analysis', 'Response_reasoning']:
        if col not in df_final.columns:
            df_final[col] = None
    
    # Process rows in parallel
    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for idx, row in df_final.iterrows():
            futures.append(executor.submit(process_row, idx, row, model_name, temperature, llm_pipeline))

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            idx, code_2, output_2, analysis_2, reasoning_2 = future.result()
            df_final.at[idx, 'Response_code'] = code_2
            df_final.at[idx, 'Response_output'] = output_2
            df_final.at[idx, 'Response_analysis'] = analysis_2
            df_final.at[idx, 'Response_reasoning'] = reasoning_2
    
    # Save the processed data with model name and UUID
    file_name, filepath = save_processed_data(df_final, model_name)
    
    return {
        "status": "success",
        "file_name": file_name,
        "message": f"Processed {len(df_final)} rows and saved to temp/processed_data"
    }
    

def run_dataset(provider: str, model_name: str, temperature : float)-> str:
    
    
    config = load_config_from_env(provider)
    
    # Allow user to override model name
    if model_name:
        config.model_name = model_name
        model_name = model_name
    else:
        model_name = config.model_name
    
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")

    pipeline = LLMPipeline(config)
    
    # Get response for all datasets
    response = get_dataset_responses(
        download_data=False,
        model_name=model_name,
        temperature=temperature,  # Use the constant TEMPERATURE
        llm_pipeline=pipeline
    )
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
     # Explicitly close all matplotlib figures to prevent tkinter errors
    plt.close('all')
    return response["file_name"]



def run_single_query(provider: str, model_name: str, temperature : float, query: str, filepath:str )-> str:
    
    
    config = load_config_from_env(provider)
    
    # Allow user to override model name
    if model_name:
        config.model_name = model_name
        model_name = model_name
    else:
        model_name = config.model_name
    
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")

    pipeline = LLMPipeline(config)
    
    # Get response for single query
    response = get_single_response(
        query=query,
        model_name=model_name,
        temperature=temperature,
        filepath=filepath,
        llm_pipeline=pipeline
    )
    
    # Print results
    print("\n=== RESULTS ===")
    print(response)
     # Explicitly close all matplotlib figures to prevent tkinter errors
    plt.close('all')
    return response