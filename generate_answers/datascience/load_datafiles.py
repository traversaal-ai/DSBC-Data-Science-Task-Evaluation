import os
import requests

def download_datasets(output_folder='temp/datafiles'):

    os.makedirs(output_folder, exist_ok=True)
    
    path_mapping = {
        'AQI Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/AQI_TRAIN.csv',
        'COVID Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/COVID_TRAIN.csv',
        'INFLATION Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/INFLATION_TRAIN.csv',
        'INSURANCE Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/INSURANCE_TRAIN.csv',
        'LIFE Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/LIFE_TRAIN.csv',
        'POPULATION Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/POPULATION_TRAIN.csv',
        'POWER Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/POWER_TRAIN.csv',
        'PRODUCTION Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/PRODUCTION_TRAIN.csv',
        'SALES Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/SALES_TRAIN.csv',
        'STOCKS Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/STOCKS_TRAIN.csv',
        'WEATHER Dataset': 'https://huggingface.co/datasets/large-traversaal/Agent-Benchmarks-Data/resolve/main/WEATHER_TRAIN.csv'
    }
    
    downloaded_files = {}
    
    for dataset_name, url in path_mapping.items():
        file_name = f"{dataset_name.replace(' Dataset', '').replace(' ', '_').lower()}_dataset.csv"
        file_path = os.path.join(output_folder, file_name)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_files[dataset_name] = file_path
            print(f"Downloaded {file_name} to {output_folder}")
            
        except Exception as e:
            print(f"Failed to download {file_name}: {str(e)}")
    
    print(f"\nTotal files downloaded: {len(downloaded_files)}/{len(path_mapping)}")
    return downloaded_files
