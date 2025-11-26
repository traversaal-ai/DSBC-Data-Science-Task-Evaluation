import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    shape = df.shape
    n_rows = shape[0]
    n_columns = shape[1]

    column_dict = {col: str(df.dtypes[col]) for col in df.columns}

    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
    if datetime_cols:
        freq_dict = {}
        for col in datetime_cols:
            series = df[col].dropna().sort_values()
            freq = pd.infer_freq(series) or 'Not Applicable'
            freq_dict[col] = freq
        frequency = freq_dict
    else:
        frequency = 'Not Applicable'

    sample_rows = df.head(10).to_dict('records')

    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = {}
    if numeric_cols:
        for col in numeric_cols:
            numeric_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': round(float(df[col].mean()), 2),
                'median': float(df[col].median())
            }

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_info = {}
    if categorical_cols:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 50:
                categorical_info[col] = {
                    'unique_count': unique_count,
                    'top_values': df[col].value_counts().head(10).to_dict()
                }
            else:
                categorical_info[col] = {
                    'unique_count': unique_count,
                    'sample_values': df[col].dropna().unique()[:10].tolist()
                }

    date_ranges = {}
    if datetime_cols:
        for col in datetime_cols:
            date_ranges[col] = {
                'min': str(df[col].min()),
                'max': str(df[col].max())
            }

    return {
        'number_of_rows': n_rows,
        'number_of_columns': n_columns,
        'columns': column_dict,
        'sample_rows': sample_rows,
        'frequency': frequency,
        'null_counts': null_counts,
        'numeric_stats': numeric_stats,
        'categorical_info': categorical_info,
        'date_ranges': date_ranges
    }