import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import pickle
import random
import seaborn as sns
import matplotlib.colors as mcolors
from tqdm import tqdm
import h5py
from itertools import groupby
from pathlib import Path

# ## set ppseq file
def find_example_file(PP_PATH, example = '178_1_7'):
    for file_ in os.listdir(PP_PATH):
        if example in file_:
            file = file_
    return file 

def convolve_movmean(y,N):
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth

def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def merge_short_blocks(lst,threshold):
    merged_list = []
    current_block = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1]:
            current_block.append(lst[i])
        else:
            if len(current_block) < threshold:
                if len(merged_list) > 0:
                    # Join with smallest adjacent block
                    if len(current_block) <= len(merged_list[-1]):
                        merged_list[-1].extend(current_block)
                    else:
                        merged_list.append(current_block + merged_list.pop())
                else:
                    # No adjacent block, discard current block
                    pass
            else:
                merged_list.append(current_block)
            
            current_block = [lst[i]]

    # Check the last block
    if len(current_block) < 3:
        if len(merged_list) > 0:
            if len(current_block) <= len(merged_list[-1]):
                merged_list[-1].extend(current_block)
            else:
                merged_list.append(current_block + merged_list.pop())
        else:
            pass
    else:
        merged_list.append(current_block)
        
    out = []
    for item in merged_list:
        out += [item[0]] * len(item)
        

    return out

def relabel_list(lst):
    # Create a dictionary to map original values to new labels
    label_dict = {}
    new_label = 1

    # Iterate over the list and assign new labels
    relabeled_list = []
    for value in lst:
        if value not in label_dict:
            label_dict[value] = new_label
            new_label += 1
        relabeled_list.append(label_dict[value])

    return relabeled_list


def most_common(lst):
    return max(set(lst), key=lst.count)

def extract_hyperdataset_details(runs_dataset):
    # Here we can see the params available in the dataset
    for key_here in runs_dataset[1][0]["params"]:
        if key_here != 'warp_type':
            params = [dataset[0]["params"][key_here] for dataset in runs_dataset]
            if len(np.unique(params)) > 1:
                print(f"{key_here}, {np.unique(params)}")
                
    # (Number of threads just controls how many threads are going on in the computation, i.e. doesn't change the model!)
    swept_params = ["Conc Param", "Seq Event Factor", "Neuron Offset Pseudo Obvs", "Fudge Factor", "Num Sequence Types"]
    chosen_params = [0.6, 1.0, 0.5, 0.1, 6]
    mean_event_proportionality = 234.41132467

    dataset_details = np.zeros([len(runs_dataset), 8])
    chosen_indices = []
    averaging_window = 50
    cutoff_likelihood = 6
    params_check = ["neuron_response_conc_param", "seq_event_rate", "neuron_offset_pseudo_obs", "mean_event_amplitude", "num_sequence_types"]



    for (dataset_id, dataset) in enumerate(runs_dataset):
        test_likelihoods = []
        for repeat in range(len(dataset)):
            if np.mean(dataset[repeat]["test_log_p"].to_numpy()[-averaging_window:]) > cutoff_likelihood:
                test_likelihoods.append(dataset[repeat]["test_log_p"].to_numpy()[-averaging_window:])
                dataset_details[dataset_id,7] += 1
            
        dataset_details[dataset_id,5] = np.mean(test_likelihoods)
        stds_per_t = np.zeros([averaging_window])
        for t in range(averaging_window):
            stds_per_t[t] = np.std([test_likelihoods[i][t] for i in range(len(test_likelihoods))])
        dataset_details[dataset_id,6] = np.mean(stds_per_t)
            
        params = [np.round(dataset[repeat]["params"][i],10) for i in params_check]    
        params[3] = np.round(params[3]/mean_event_proportionality, 3)
        params[1] = 5*params[1]/params[4]
        params = np.array(params)
        dataset_details[dataset_id,:5] = params
        

    cleaned_dataset_details = np.zeros(dataset_details.shape)
    counter = 0
    for dataset_detail in dataset_details:
        if np.logical_not(np.isnan(dataset_detail[5])):
            cleaned_dataset_details[counter, :] = dataset_detail
            
            if np.sum(dataset_detail[:5] == chosen_params) == 5:
                chosen_indices.append(counter)
            
            counter += 1
    dataset_details = cleaned_dataset_details[:counter,:]
    
    return dataset_details


def add_arrows(i, this_param_vals, means_here, stds_here):
    if i == 0:
        arrow_x = 6
    elif i == 1:
        arrow_x = 0
    elif i == 2:
        arrow_x = -1.2
    elif i == 3:
        arrow_x = 0
    elif i == 4:
        arrow_x = -0.7
    else:
        return

    # find nearest data point
    x_vals = np.log(this_param_vals) if i > 0 else this_param_vals
    nearest_idx = np.argmin(np.abs(x_vals - arrow_x))
    arrow_y = means_here[nearest_idx]

    # draw arrow with FIXED visual length
    plt.annotate(
        '',
        xy=(arrow_x, arrow_y),            # arrow tip
        xytext=(0, 20),                    # arrow base offset (20 points upward)
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red', lw=1)
    )


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
