import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
import h5py
import ast
from scipy import stats 

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

    with h5py.File(filename, 'r') as f:
        return {key: load_item(f[key]) for key in f.keys()}