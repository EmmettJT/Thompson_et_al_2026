import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ast
import seaborn as sns
import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import wilcoxon, mannwhitneyu
import re
import scipy
from tqdm import tqdm 
import random
import ast
from pathlib import Path
import h5py

def extract_errors_into_df(input_):

    correct = []
    error = []
    other = []

    for i in range (3):
        for e in range(3):
            if e == 0:
                correct = correct + [list(np.array(input_[i])[:,e])]
            if e == 1:
                error = error + [list(np.array(input_[i])[:,e])]
            if e == 2:
                other = other + [list(np.array(input_[i])[:,e])]


    errors = pd.DataFrame({ 'Group':['Early','Pre', 'Post'], 'correct':  correct, 
        'error': error, 'other': other })
    return errors 

def average_across_animals_transition_matrix(AA_transits):
    group_means = []
    for group in AA_transits:
        AA_data= []
        for animal in group:
            flat_list = [item for sublist in animal for item in sublist]
            normalised = list(np.array(flat_list) / max(flat_list))
            AA_data = AA_data + [normalised]
        concat_AA_data = conactinate_nth_items(AA_data)

        # recreate transition matrix: 
        means= [[]]*8
        count = 0
        index = 0
        for item in concat_AA_data:
            means[index] = means[index] + [np.mean(item)]
            count = count + 1
            if count == 4:
                count = 0
                index = index + 1

        group_means = group_means + [means]
    return(group_means)

def normalise_transition_matrix_means(group_means):
    normalised_means = []
    for e_type in group_means:

        transposed_data  = np.array(e_type).T.tolist()

        AA_data= []
        flat_list = []
        for s_port in transposed_data:
            normalised = list(np.array(s_port) / sum(s_port))
            flat_list = flat_list + normalised

            AA_data = AA_data + [flat_list]
        concat_AA_data = conactinate_nth_items(AA_data)

        # recreate transition matrix: 
        means= [[]]*4
        count = 0
        index = 0
        for item in concat_AA_data:
            means[index] = means[index] + [np.mean(item)]
            count = count + 1
            if count == 8:
                count = 0
                index = index + 1

        normalised_means = normalised_means + [np.array(means).T.tolist()]
        
    return normalised_means

def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def parse_error_list(x):
    if isinstance(x, str):
        # extract all floats using regex
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", x)
        return np.array([float(n) for n in nums])
    return x  # already an array/list

def convolve_movmean(y,N):
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth


def parse_training_levels(training_levels):
    t_levels = []
    for row in training_levels:
        row = row.replace('nan', 'None')
        row = '[' + ', '.join(convert_float_string(x) for x in row.strip('[]').split(', ')) + ']'
        t_levels.append(ast.literal_eval(row))
    return t_levels

def convert_float_string(s):
    try:
        # Attempt to convert scientific notation to a plain float string
        if 'e' in s.lower():
            value = float(s)
            return str(value)
        else:
            return s  # Return original string if not in scientific notation
    except ValueError:
        return s  # Return original string if not a valid float

def calculate_mean_std(t_levels, mask):
    """
    Selects the trial lists from t_levels using a boolean mask,
    concatenates nth items, and computes mean and std curves.
    Handles ragged input safely by avoiding NumPy masking.
    """
    # Safe boolean masking without NumPy
    selected = [t for t, m in zip(t_levels, mask) if m]
    # Combine the nth items from each selected trial list
    trial_scores = conactinate_nth_items(selected)
    # Compute mean and std across trials for each trial index
    mean_curve = [np.mean(scores) for scores in trial_scores]
    std_curve  = [np.std(scores)  for scores in trial_scores]

    return mean_curve, std_curve

def fill_between_mean_std(ax, mean_curve, std_curve, color,xlim):
    upper = np.array(mean_curve[:xlim]) + np.array(std_curve[:xlim])
    lower = np.array(mean_curve[:xlim]) - np.array(std_curve[:xlim])
    upper[upper > 50] = 50  # Ceiling effect cutoff
    ax.fill_between(range(len(upper)), lower, upper, alpha=0.2, edgecolor='None', facecolor=color, linewidth=1, linestyle='dashdot', antialiased=True)
    
def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def convolve_movmean(y,N):
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth

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