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
from pathlib import Path
import ast 
from itertools import groupby

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


def load_H5_bodypart(tracking_path,video_type, tracking_point):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file:
            if 'task' in file:
                back_file = pd.read_hdf(tracking_path + file)     
                
    # drag data out of the df
    scorer = back_file.columns.tolist()[0][0]
    body_part = back_file[scorer][tracking_point]
    
    parts=[]
    for item in list(back_file[scorer]):
        parts+=[item[0]]
    print(np.unique(parts))
    
    # clean and interpolate frames with less than 98% confidence
    clean_and_interpolate(body_part,0.98)
    
    return(body_part)
  
def load_H5_ports(tracking_path,video_type):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file:
            if 'port' in file:
                back_ports_file = pd.read_hdf(tracking_path + file)

    ## same for the ports:
    scorer = back_ports_file.columns.tolist()[0][0]
        
    if video_type == 'back':
        port1 =back_ports_file[scorer]['port2']
        port2 =back_ports_file[scorer]['port1']
        port3 =back_ports_file[scorer]['port6']
        port4 =back_ports_file[scorer]['port3']
        port5 =back_ports_file[scorer]['port7']
    else:
        port1 =back_ports_file[scorer]['Port2']
        port2 =back_ports_file[scorer]['Port1']
        port3 =back_ports_file[scorer]['Port6']
        port4 =back_ports_file[scorer]['Port3']
        port5 =back_ports_file[scorer]['Port7']

    clean_and_interpolate(port1,0.98)
    clean_and_interpolate(port2,0.98)
    clean_and_interpolate(port3,0.98)
    clean_and_interpolate(port4,0.98)
    clean_and_interpolate(port5,0.98)
    
    return(port1,port2,port3,port4,port5)

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def clean_and_interpolate(head_centre,threshold):
    bad_confidence_inds = np.where(head_centre.likelihood.values<threshold)[0]
    newx = head_centre.x.values
    newx[bad_confidence_inds] = 0
    newy = head_centre.y.values
    newy[bad_confidence_inds] = 0

    start_value_cleanup(newx)
    interped_x = interp_0_coords(newx)

    start_value_cleanup(newy)
    interped_y = interp_0_coords(newy)
    
    head_centre['interped_x'] = interped_x
    head_centre['interped_y'] = interped_y
    
def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0 
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        working = 0
        if value > 0:
            start_value = value
            start_index = index
            working = 1
            break
    if working == 1:
        for x in range(start_index):
            coords[x] = start_value
            
def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
#                         print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
#     print('function exiting')
    return(coords_list)

def sortperm_neurons(bkgd_log_proportions_array,config,neuron_response_df, sequence_ordering=None, th=0.2):
    ## this is number of neurons in total
    N_neurons= bkgd_log_proportions_array.shape[1]
    ## number of sequences from json file 
    n_sequences = config["num_sequence_types"]
    # the 18 neuron params for each neuron from the last iteration
    all_final_globals = neuron_response_df.iloc[-N_neurons:]
    # this cuts it down to just the first 6 params - i think this correspond sto the first param for each seq type? response probABILITY - ie the chance that a neuron spikes in a given latent seq 
    resp_prop = np.exp(all_final_globals.values[:, :n_sequences])#
    # this takes the next 6 params - which i think are the offset values
    offset = all_final_globals.values[-N_neurons:, n_sequences:2*n_sequences]
    ## finds the max response value - ie. which seq it fits to? 
    peak_response = np.amax(resp_prop, axis=1)
    # then threshold the reponse
    has_response = peak_response > np.quantile(peak_response, th)
    # I thin this is the sequence that the neuron has the max response for: ie. we are ordering them by max response 
    preferred_type = np.argmax(resp_prop, axis=1)
    if sequence_ordering is None:
        # order them by max reponse 
        ordered_preferred_type = preferred_type
    else:
        #order them differnetly 
        ordered_preferred_type = np.zeros(N_neurons)#
        # loop through each sequence
        for seq in range(n_sequences):
            # where does  max repsone = user defined seque
            seq_indices = np.where(preferred_type == sequence_ordering[seq])
            # change order to different seq
            ordered_preferred_type[seq_indices] = seq

    # reorder the offset params according to max respsone
    preferred_delay = offset[np.arange(N_neurons), preferred_type]
    Z = np.stack([has_response, ordered_preferred_type+1, preferred_delay], axis=1)
    indexes = np.lexsort((Z[:, 2], Z[:, 1], Z[:, 0]))
    return indexes,ordered_preferred_type

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

def split_list(nums):
    sublists = []
    current_sublist = [nums[0]]
    current_element = nums[0]
    for i in range(1,len(nums)):
        if nums[i] == current_element:
            current_sublist.append(nums[i])
        else:
            sublists.append(current_sublist)
            current_sublist = [nums[i]]
            current_element = nums[i]
    sublists.append(current_sublist)
    return sublists

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



def load_pickle(file_path):
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)

def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def load_tracking_data(tracking_path):
    back_head_centre = load_H5_bodypart(tracking_path, 'back', 'head_centre')
    back_p1, back_p2, back_p3, back_p4, back_p5 = load_H5_ports(tracking_path, 'back')
    return back_head_centre, (back_p1, back_p2, back_p3, back_p4, back_p5)

def load_processed_spike_data(pp_path, file):
    print("\nLOADING processed_spike_data")
    analysis_path = os.path.join(pp_path, file, "analysis_output")
    
    latent_event_history_df_split = load_pickle(os.path.join(analysis_path, "latent_event_history_df_split.pickle"))
    spikes_seq_type_adjusted = load_pickle(os.path.join(analysis_path, "spikes_seq_type_adjusted.pickle"))
    neuron_order = np.load(os.path.join(analysis_path, 'neuron_order.npy'))
    ordered_preferred_type = np.load(os.path.join(analysis_path, 'ordered_preferred_type.npy'))
    neuron_index = np.load(os.path.join(analysis_path, 'neuron_index.npy'))

    config = eval(load_json(os.path.join(pp_path, file, 'config_file.json')))

    return latent_event_history_df_split, spikes_seq_type_adjusted, neuron_order, ordered_preferred_type, neuron_index, config

def load_input_params(pp_path, file, mouse_session_recording):
    input_params_path = os.path.join(pp_path, file, 'trainingData', f'params_{mouse_session_recording}.json')
    input_config = load_json(input_params_path)
    behav_time_interval_start = input_config['time_span'][0]
    print(f"      A corresponding time span has been found. Time span set to {behav_time_interval_start}")
    return behav_time_interval_start

def load_behav_data(dat_path, behav_time_interval_start):
    print("\nLOADING BEHAV DATA")
    behav_sync = pd.read_csv(os.path.join(dat_path, 'behav_sync', '2_task', 'Behav_Ephys_Camera_Sync.csv'))
    transitions = pd.read_csv(os.path.join(dat_path, 'behav_sync', '2_task', 'Transition_data_sync.csv'))

    behav_mask = (behav_sync.PokeIN_EphysTime > behav_time_interval_start[0]) & (behav_sync.PokeIN_EphysTime < behav_time_interval_start[1])
    poke_in_times = behav_sync[behav_mask].PokeIN_EphysTime - behav_time_interval_start[0]
    ports = behav_sync[behav_mask].Port
    print('done')
    return behav_sync, transitions, poke_in_times, ports,behav_mask

def Load_example_data(pp_path, file, tracking_path, dat_path, mouse_session_recording = '178_1_7'):
    # Load processed spike data
    latent_event_history_df_split, spikes_seq_type_adjusted, neuron_order, ordered_preferred_type, neuron_index, config = load_processed_spike_data(pp_path, file)
    
    # Load DLC tracking data
    print("\nLOADING DLC TRACKING DATA")
    back_head_centre, back_ports = load_tracking_data(tracking_path)
    
    # Load the timespan used for pppseq
    behav_time_interval_start = load_input_params(pp_path, file, mouse_session_recording)
    
    # Load behaviour data
    behav_sync, transitions, poke_in_times, ports,behav_mask = load_behav_data(dat_path, behav_time_interval_start)
    
    
    neuron_response_df = pd.read_csv(pp_path + file + r"\neuron_response.csv")
    bkgd_log_proportions_array = pd.read_csv(pp_path + file + r"\bkgd_log_proportions_array.csv")

    # Return all loaded data for further processing if needed
    return {
        "latent_event_history_df_split": latent_event_history_df_split,
        "spikes_seq_type_adjusted": spikes_seq_type_adjusted,
        "neuron_order": neuron_order,
        "ordered_preferred_type": ordered_preferred_type,
        "neuron_index": neuron_index,
        "config": config,
        "back_head_centre": back_head_centre,
        "back_ports": back_ports,
        "behav_sync": behav_sync,
        "transitions": transitions,
        "poke_in_times": poke_in_times,
        "ports": ports,
        "neuron_response_df" : neuron_response_df,
        "bkgd_log_proportions_array" : bkgd_log_proportions_array,
        "behav_time_interval_start" : behav_time_interval_start,
        "behav_mask" : behav_mask
    }


def shuffle(aList):
    random.shuffle(aList)
    return aList

### Plot sequences - basic
def plot_unordered_ordered_colored_example(data, ordering,colors, timeframe = [200,230]):

    timeframe = [200,230]

    mask = (data["spikes_seq_type_adjusted"].timestamp>timeframe[0])*(data["spikes_seq_type_adjusted"].timestamp<timeframe[-1])

    ## neuron order:

    #define neuron order
    neuron_index,ordered_preferred_type = sortperm_neurons(data["bkgd_log_proportions_array"],data["config"],data["neuron_response_df"], sequence_ordering=ordering)
    # make a list of idndies for each neurons new position
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))

    neuron_permute_loc = np.array(shuffle(list(neuron_permute_loc.astype(int))))
    neuron_order = neuron_permute_loc[data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

    ## plotting:
    nrow = 3 
    ncol = 1
    fig, axs = plt.subplots(nrow, ncol,figsize=(13, 16))

    for ind, ax in enumerate(fig.axes):

        if ind == 0:
            # plot background in grey 
            background_keep_mask = data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted <= 0
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],marker = 'o', s=20, linewidth=0,color = 'k' ,alpha=1)

            # plot spikes without background
            background_remove_mask =  data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted >= 0
            c_ = np.array(colors)[ data["spikes_seq_type_adjusted"][mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
            # ## faster:
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],marker = 'o', s=20, linewidth=0,color = 'k' ,alpha=1)

        if ind == 1:
            # make a list of idndies for each neurons new position
            neuron_permute_loc = np.zeros(len(neuron_index))
            for i in range(len(neuron_index)):
                neuron_permute_loc[i] = int(list(neuron_index).index(i))
    #         neuron_order = neuron_permute_loc[unmasked_spikes_df.neuron-1]
            neuron_order = neuron_permute_loc[ data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

            # plot background in grey 
            background_keep_mask = data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted <= 0
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],marker = 'o', s=20, linewidth=0,color = 'grey' ,alpha=0.5)

            # plot spikes without background
            background_remove_mask =  data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted >= 0
            c_ = np.array(colors)[ data["spikes_seq_type_adjusted"][mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
            # ## faster:
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],marker = 'o', s=20, linewidth=0,color = 'k' ,alpha=1)

        if ind == 2:

            # plot background in grey 
            background_keep_mask =  data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted <= 0
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],marker = 'o', s=20, linewidth=0,color = 'grey' ,alpha=0.3)

            # plot spikes without background
            background_remove_mask = data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted >= 0
            c_ = np.array(colors)[data["spikes_seq_type_adjusted"][mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
            # ## faster:
            ax.scatter( data["spikes_seq_type_adjusted"][mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],marker = 'o', s=20, linewidth=0,color = c_ ,alpha=1)
    return

def plot_zoomed_example_raster(data,ordering,colors,timeframe = [223,230]):

    timeframe = [223,230]

    mask = (data["spikes_seq_type_adjusted"].timestamp>timeframe[0])*(data["spikes_seq_type_adjusted"].timestamp<timeframe[-1])

    ## neuron order:

    #define neuron order
    neuron_index,ordered_preferred_type = sortperm_neurons(data["bkgd_log_proportions_array"],data["config"],data["neuron_response_df"], sequence_ordering=ordering)
    # make a list of idndies for each neurons new position
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))

    neuron_permute_loc = np.array(shuffle(list(neuron_permute_loc.astype(int))))
    neuron_order = neuron_permute_loc[data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

    # make a list of idndies for each neurons new position
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))
    #         neuron_order = neuron_permute_loc[unmasked_spikes_df.neuron-1]
    neuron_order = neuron_permute_loc[ data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

    ## plotting:
    nrow = 1
    ncol = 1

    fig, ax = plt.subplots(nrow, ncol,figsize=(9, 5))

    # plot background in grey 
    background_keep_mask =  data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted <= 0
    ax.scatter( data["spikes_seq_type_adjusted"][mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],marker = 'o', s=20, linewidth=0,color = 'grey' ,alpha=0.3)

    # plot spikes without background
    background_remove_mask = data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted >= 0
    c_ = np.array(colors)[data["spikes_seq_type_adjusted"][mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
    # ## faster:
    ax.scatter( data["spikes_seq_type_adjusted"][mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],marker = 'o', s=20, linewidth=0,color = c_ ,alpha=1)

    return

def plot_tracking_aligned(data,strt_,end_,events,start_ts,end_ts, colors,max_index):
    nrow = 1 
    ncol = 1
    fig, axs = plt.subplots(nrow, ncol,figsize=(15, 8))

    for ind, ax in enumerate(fig.axes):
        for i in range(5):
            ax.plot(data["port_centroids"].x[i],data["port_centroids"].y[i],'o',color = 'grey', markersize = 125)
        for i in range(int(events)):
            ax.plot(data["back_head_centre"]['interped_x'].values[int(start_ts[i])-1:int(end_ts[i])+1],data["back_head_centre"]['interped_y'].values[int(start_ts[i])-1:int(end_ts[i])+1],'-',color =  np.array(colors)[np.array(max_index)+1][strt_:end_][i], alpha = 1)

        min_x = data["port_centroids"].x[1] - (data["port_centroids"].x[0] - data["port_centroids"].x[1])
        max_x = data["port_centroids"].x[3] + (data["port_centroids"].x[0] - data["port_centroids"].x[1])
        min_y = data["port_centroids"].y[2] - (data["port_centroids"].y[0] - data["port_centroids"].y[2])
        max_y = data["port_centroids"].y[0] + (data["port_centroids"].y[0] - data["port_centroids"].y[2])
        
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)

        ax.invert_yaxis()
    return


def remove_neighboring_duplicates(nums):
    new_list = []
    for i in range(len(nums)):
        if i == 0 or nums[i] != nums[i-1]:
            new_list.append(nums[i])
    return new_list

def create_pairs(nums):
    return [nums[i]*10+nums[i+1] for i in range(len(nums)-1)]

def determineTransitionNumber(TimeFiltered_seqs,possible_transitions):
    TransitionTypesIndex = possible_transitions
    trajects = []
    for ind, transits in enumerate(TimeFiltered_seqs):
        trajects = np.append(trajects,transits)
    transition_number = []
    for transit_types in TransitionTypesIndex:
        temp = (np.where(trajects == float(transit_types)))
        transition_number.append(len(temp[0]))
    return transition_number


def return_inds_for_seq_groups(lst):
    groups = []
    new = True
    for ind,item in enumerate(lst):
        if new:
            if item > 0:
                start = ind
                new = False
        else:
            if item == 0:
                end = ind-1
                groups.append((start, end))
                new = True
    return groups

def calculate_percentages(numbers):
    total = sum(numbers)
    percentages = [(number/total) * 100 for number in numbers]
    return percentages


def reorder_lists(original_list, index_list):
    return [original_list[i] for i in index_list]

def add_missing_numbers(lst, start_num, end_num):
    new_lst = lst.copy()
    missing_numbers = [num for num in range(start_num, end_num+1) if num not in lst]
    new_lst += missing_numbers
    return new_lst,len(missing_numbers)

def calculate_mean_sem_of_neuron_seq_repsonses(resp_prop_df, var_arg ):

    all_mean_mean_resp_prop = []
    all_sem_mean_resp_prop = []
    all_index =[]

    for i_ in range(len(resp_prop_df)):
        row = resp_prop_df.loc[resp_prop_df.index == i_]

        mean_mean_resp_prop = []
        sem_mean_resp_prop = []
        for item in list(row)[1::]:
            if not var_arg == 'offset':
                resp_prop_unlog = np.exp(list(row[item])[0])
            else:
                resp_prop_unlog = list(row[item])[0]
            mean_mean_resp_prop += [np.mean(resp_prop_unlog)]
            if var_arg == 'offset':
                sem_mean_resp_prop += [scipy.stats.sem(resp_prop_unlog)[0]]
            else:
                sem_mean_resp_prop += [scipy.stats.sem(resp_prop_unlog)]

        all_mean_mean_resp_prop += [mean_mean_resp_prop]
        all_sem_mean_resp_prop += [sem_mean_resp_prop]
        all_index+=[i_]*len(mean_mean_resp_prop)

    return all_mean_mean_resp_prop,all_sem_mean_resp_prop,all_index

def create_color_gradient(start_color, end_color, num_samples):
    # Create a linear gradient between start_color and end_color
    gradient = np.linspace(0, 1, num_samples)

    # Create a color map using the start_color and end_color
    color_map = mcolors.LinearSegmentedColormap.from_list(
        'color_gradient', [start_color, end_color]
    )

    # Generate a list of colors from the color map
    colors = [color_map(gradient_value) for gradient_value in gradient]

    return colors

###### determine gaps between each:
def find_sequence(long_sequence, short_sequence):
    long_length = len(long_sequence)
    short_length = len(short_sequence)  
    occurrences = []
    for i in range(long_length - short_length + 1):
        if long_sequence[i:i+short_length] == short_sequence:
            occurrences.append(i)
    return occurrences

# ## set ppseq file
def find_example_file(PP_PATH, example = '178_1_7'):
    for file_ in os.listdir(PP_PATH):
        if example in file_:
            file = file_
    return file 