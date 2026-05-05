import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
from pathlib import Path

def load_h5(filename):
    """
    Load a dictionary saved in HDF5, recursively reconstructing:
    - DataFrames with proper columns
    - Nested dictionaries
    - Lists (including 1D arrays)
    - Bytes decoded to Python strings
    - Numeric strings converted to floats
    """
    def decode_and_convert(x):
        # Decode bytes
        if isinstance(x, bytes):
            x = x.decode('utf-8')
        # Convert numeric strings to float if possible
        if isinstance(x, str):
            try:
                x_float = float(x)
                return x_float
            except ValueError:
                return x
        return x

    def load_item(group):
        if isinstance(group, h5py.Group):
            # DataFrame detection
            if 'data' in group and 'columns' in group.attrs:
                columns = [col.decode('utf-8') for col in group.attrs['columns']]
                data = group['data'][()]
                # Decode bytes and convert numeric strings
                data = np.array([[decode_and_convert(cell) for cell in row] for row in data])
                return pd.DataFrame(data, columns=columns)
            else:
                # Nested dictionary
                return {key: load_item(group[key]) for key in group.keys()}
        elif isinstance(group, h5py.Dataset):
            data = group[()]
            # Decode bytes and convert numeric strings
            if data.dtype.kind in {'S', 'O'}:
                data = np.array([decode_and_convert(x) for x in data.flat])
                if data.ndim == 1:
                    return data.tolist()  # Convert 1D arrays to list
                return data.reshape(group.shape)
            else:
                return data
        else:
            return group

    with h5py.File(filename, 'r') as f:
        return {key: load_item(f[key]) for key in f.keys()}
    
def convert_loaded_data(data):
    """
    Recursively convert HDF5-loaded data to Python-friendly types:
    - Numeric strings -> floats
    - 1D arrays -> lists
    - Leaves strings or DataFrames as-is
    """
    if isinstance(data, np.ndarray):
        if data.dtype.kind in {'U', 'S', 'O'}:
            # Try to convert each element to float, fallback to string
            def try_float(x):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return x
            data = np.vectorize(try_float)(data)
        # Convert 1D arrays to list
        if data.ndim == 1:
            return data.tolist()
        return data
    elif isinstance(data, pd.DataFrame):
        # Apply conversion to all DataFrame cells
        return data.applymap(lambda x: float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else x)
    elif isinstance(data, dict):
        return {k: convert_loaded_data(v) for k, v in data.items()}
    else:
        return data