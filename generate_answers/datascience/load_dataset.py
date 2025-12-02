from datasets import load_dataset
import pandas as pd
from typing import Dict, Optional

def load_dataset_from_hf() -> pd.DataFrame:
    dataset = load_dataset("large-traversaal/dsbc_v0")
    df = dataset['train'].to_pandas()
    print(f"Dataset loaded! Shape: {df.shape}")
    return df

def get_dataset_path_mapping() -> pd.DataFrame:
    df = load_dataset_from_hf()
    
    path_mapping: Dict[str, str] = {
        'AQI Dataset': 'temp/datafiles/aqi_dataset.csv',
        'COVID Dataset': 'temp/datafiles/covid_dataset.csv',
        'INFLATION Dataset': 'temp/datafiles/inflation_dataset.csv',
        'INSURANCE Dataset': 'temp/datafiles/insurance_dataset.csv',
        'LIFE Dataset': 'temp/datafiles/life_dataset.csv',
        'POPULATION Dataset': 'temp/datafiles/population_dataset.csv',
        'POWER Dataset': 'temp/datafiles/power_dataset.csv',
        'PRODUCTION Dataset': 'temp/datafiles/production_dataset.csv',
        'SALES Dataset': 'temp/datafiles/sales_dataset.csv',
        'STOCKS Dataset': 'temp/datafiles/stocks_dataset.csv',
        'WEATHER Dataset': 'temp/datafiles/weather_dataset.csv'
    }
    
    df['PATH'] = df['Dataset'].map(path_mapping)

    return df
    