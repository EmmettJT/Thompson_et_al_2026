import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.interpolate import interp1d
import math
from scipy.spatial import KDTree
import seaborn as sns;
import pickle

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
            if 'port' in file or 'PORT' in file:
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


# Function to find corresponding number in another column
def find_corresponding(nums,df_dict):
    return [df_dict[num] for num in nums]

def SaveFig(file_name,figure_dir):
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir + file_name, bbox_inches='tight')
#     plt.show()
    plt.close()
    
    
def load_in_paths(pp_file, PP_PATH, DAT_PATH,run_index,ignore_list):
    mir = '_'.join(pp_file.split('_')[0:3])
    print(str(run_index+1) + '/' + str(len(os.listdir(PP_PATH))-1) + '-------------------------------------------------------------------------')
    print(pp_file)
    mouse_session_recording = pp_file.split('_')[0] + '_' + pp_file.split('_')[1] + '_' + pp_file.split('_')[2] 
    skip = False
    for item in ignore_list:
        if item == mouse_session_recording:
            skip = True

    ## set dat_path:
    for file_ in os.listdir(DAT_PATH):
        if mouse_session_recording.split('_')[0] in file_:
            if mouse_session_recording.split('_')[1] == file_[-1]:
                dat_path = os.path.join(DAT_PATH,file_)
    for recording in os.listdir(os.path.join(DAT_PATH,dat_path)):
        if recording.split('_')[0][9::] == mouse_session_recording.split('_')[-1]:
            dat_path = os.path.join(dat_path,recording)

    # set tracking path
    for file_ in os.listdir(dat_path + r"\video\tracking\\"):
        if 'EJT' in mir:
            if 'task' in file_:
                if not 'clock' in file_:
                    tracking_path = os.path.join(dat_path + r"\video\tracking\\",file_) + '\\'
        else:
            tracking_path = dat_path + r"\video\tracking\\"
            

    return mir,mouse_session_recording,tracking_path,dat_path

                    
def load_PPSEQ_data(PP_PATH,pp_file,mouse_session_recording):

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    ## LOAD 
    print("LOADING PPSEQ DATA")
    print('\n')
    #The assignment history frame (assigment_hist_frame.csv): Spikes by iterations, how each spike is assigned to a sequence ID (in latent_event_hist) or to background (-1)
    assignment_history_df = pd.read_csv(PP_PATH + pp_file + r"\assigment_hist_frame.csv")

    # latent_event_hist.csv: history of latent events. All latent events across all iterations have a row
    latent_event_history_df = pd.read_csv(PP_PATH + pp_file + r"\latent_event_hist.csv")

    # seq_type_log_proportions: log p of each type of sequence at each iteration
    seq_type_log_proportions_df = pd.read_csv(PP_PATH + pp_file + r"\seq_type_log_proportions.csv")

    # neuron_responses.csv: iterations x neurons by 3(number of sequences). Each neuron has three parameters per sequence to describe how it is influenced by each sequence type. 
    # Each iteration these are resampled, therefore there are number of neurons by iterations by 3 by number of sequences of these numbers.
    neuron_response_df = pd.read_csv(PP_PATH + pp_file + r"\neuron_response.csv")

    masking = False
    for dat_files in os.listdir(PP_PATH + pp_file):
        if 'unmasked_spikes' in dat_files:
            masking = True
            print('masking was used')

    if masking == True:
        #log_p_hist.csv: the history of the log_p of the model
        log_p_hist_df = pd.read_csv(PP_PATH + pp_file + r"\test_log_p_hist.csv")

        unmasked_spikes_df = pd.read_csv(PP_PATH + pp_file + r"\unmasked_spikes.csv")
    else:
        log_p_hist_df = pd.read_csv(PP_PATH + pp_file + r"\log_p_hist.csv")

        spikes_file = os.path.join(PP_PATH + pp_file,'trainingData\\') + mouse_session_recording + '.txt'
        neuron_ids, spike_times= [], []
        with open(spikes_file) as f:
            for (i, line) in enumerate(f.readlines()):
                neuron_id, spike_time = line.split('\t')
                spike_time = float(spike_time.strip())
                neuron_id = float(neuron_id)
                spike_times.append(spike_time)
                neuron_ids.append(neuron_id)
        unmasked_spikes_df = pd.DataFrame({'neuron':neuron_ids,'timestamp':spike_times}) 
        bkgd_log_proportions_array = pd.read_csv(PP_PATH + pp_file + r"\bkgd_log_proportions_array.csv")

    # Opening JSON file
    f = open(PP_PATH + pp_file + r'\config_file.json')
    # returns JSON object as a dictionary
    config = eval(json.load(f))
    print(f'      done')

    ## LOAD behaviour data
    print('\n')
    print("LOADING BEHAV DATA")

    ## load in the timespan used for pppseq:
    input_params_path = os.path.join(PP_PATH + pp_file,'trainingData\\') + ('params_' + mouse_session_recording +'.json')
    # Opening JSON file
    f = open(input_params_path)
    # returns JSON object as 
    # a dictionary
    input_config = json.load(f)
    behav_time_interval_start = input_config['time_span']
    print(f"      A corresponding time span has been found. Time span set to {behav_time_interval_start}")
            
    return assignment_history_df,latent_event_history_df,seq_type_log_proportions_df,neuron_response_df,log_p_hist_df,unmasked_spikes_df,bkgd_log_proportions_array,behav_time_interval_start

def plot_save_log_l_curve(log_p_hist_df):
    # find 95% of growth value and when it crossed this
    max_ = max(log_p_hist_df.x1)
    min_ = min(log_p_hist_df.x1)
    growth = max_ - min_
    _prcntile =  max_ - (0.02 * growth)

    ## model log likley hood curve
    plt.plot(log_p_hist_df.x1)
    plt.axhline(y=_prcntile, color='r', linestyle='--')
    plt.show()

def filter_cluster_chunks(
    r_seq_type,
    r_start_,
    r_end_,
    chunk_df,
    neuron_order,
    chunk_mask,
    colors=None,
    ax=None,
):

    filtered_cluster_sequence_type = []
    filtered_cluster_timestamps = []
    filtered_num_spikes = []
    filtered_num_neurons = []
    filtered_first_spike_time = []
    filtered_last_spike_time = []
    filtered_cluster_neurons = []
    event_length = []
    cluster_neuron_order = []
    min_spikes_filter = 7
    min_neurons_involved_filter = 5
    
    for i in range(len(r_seq_type)):
        mask = (chunk_df.timestamp >= r_start_[i]) & (chunk_df.timestamp <= r_end_[i])

        r_event_df = chunk_df[mask].copy()
        neuron_orders = neuron_order[chunk_mask][mask]

        # keep only neurons of the current sequence type
        seq_mask = r_event_df.sequence_type_adjusted == r_seq_type[i]
        r_event_df = r_event_df[seq_mask]
        neuron_orders = neuron_orders[seq_mask]

        if len(r_event_df) == 0:
            continue

        num_spikes = len(r_event_df)
        num_neurons = r_event_df.neuron.nunique()
        first_spike = r_event_df.timestamp.min()
        last_spike = r_event_df.timestamp.max()

        if num_spikes < min_spikes_filter:
            continue

        if num_neurons < min_neurons_involved_filter:
            continue

        # optional plotting
        if ax is not None and colors is not None:
            ax.axvspan(
                first_spike,
                last_spike,
                color=colors[r_seq_type[i]],
                alpha=0.5,
            )

        # save filtered cluster info
        filtered_cluster_sequence_type.append(r_seq_type[i])
        filtered_cluster_timestamps.append(list(r_event_df.timestamp.values))
        filtered_cluster_neurons.append(list(r_event_df.neuron.values))
        filtered_num_spikes.append(num_spikes)
        filtered_num_neurons.append(num_neurons)
        filtered_first_spike_time.append(first_spike)
        filtered_last_spike_time.append(last_spike)
        event_length.append(last_spike - first_spike)
        cluster_neuron_order.append(neuron_orders)
        
    return pd.DataFrame({'cluster_seq_type':filtered_cluster_sequence_type, 'num_spikes':filtered_num_spikes,  'num_neurons': filtered_num_neurons,'first_spike_time':filtered_first_spike_time,'event_length':event_length,'last_spike_time':filtered_last_spike_time, 'cluster_spike_times':filtered_cluster_timestamps,'cluster_neurons':filtered_cluster_neurons,'spike_plotting_order': cluster_neuron_order})

    
def plot_data_raster(behav_time_interval_start, spikes_df, neuron_index, colors):
    # calculate interval timings and end points
    interval_lengths = []
    for interval in behav_time_interval_start:
        interval_lengths += [np.diff(interval)[0]]
    total_time = sum(interval_lengths)
    interval_end_points = np.cumsum(interval_lengths)

    # Plot sequences - basic
    timeframe = [0, total_time]
    mask = (spikes_df.timestamp > timeframe[0]) * (spikes_df.timestamp < timeframe[-1])

    # Define neuron order
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))
    neuron_order = neuron_permute_loc[(spikes_df.neuron - 1).astype(int)]

    # Plotting
    fig, [ax, ax2] = plt.subplots(2, 1, figsize=(20, 20))

    # Plot background in grey
    background_keep_mask = (spikes_df[mask].sequence_type_adjusted < 0) | (spikes_df[mask].sequence_type_adjusted >= 7.0)
    ax.scatter(spikes_df[mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],
               marker='o', s=40, linewidth=0, color='lightgrey', alpha=0.3)
    c_ = np.array(colors)[spikes_df[mask][background_keep_mask].sequence_type_adjusted.values.astype(int)]
    ax2.scatter(spikes_df[mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],
                marker='o', s=40, linewidth=0, color=c_, alpha=0.3)
    ax2.set_title('extra sequences and background only')

    # Plot spikes without background
    background_remove_mask = (spikes_df[mask].sequence_type_adjusted >= 0) * \
                             (spikes_df[mask].sequence_type_adjusted != 7.0) * \
                             (spikes_df[mask].sequence_type_adjusted != 8.0)
    c_ = np.array(colors)[spikes_df[mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
    ax.scatter(spikes_df[mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],
               marker='o', s=40, linewidth=0, color=c_, alpha=1)
    ax.set_title('held sequences in color and extra sequences + background in grey')

    for end_p in interval_end_points:
        ax.axvline(x=end_p, color='k')

    return interval_end_points,neuron_order


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

def seperate_lists(numbers):
    chunks = []
    indices = []
    current_chunk = []
    current_indices = []
    for i, num in enumerate(numbers):
        if num == 0:
            if current_chunk:
                chunks.append(current_chunk)
                indices.append(current_indices)
                current_chunk = []
                current_indices = []
        else:
            current_chunk.append(num)
            current_indices.append(i)
    if current_chunk:
        chunks.append(current_chunk)
        indices.append(current_indices)
    return chunks, indices

def cluster_events(start_times, end_times, threshold):
    clusters = []
    for i in range(len(start_times)):
        event_added = False
        for cluster in clusters:
            for index in cluster:
                if (start_times[i] <= end_times[index] + threshold and end_times[i] >= start_times[index] - threshold):
                    cluster.append(i)
                    event_added = True
                    break
            if event_added:
                break
        if not event_added:
            clusters.append([i])
    return clusters

def catagorize_seqs(real_order,num_dominant_seqs,order):
    
    #deal wih the fact that the way I order the sequence messes up the order a bit
    if not len(real_order) == num_dominant_seqs:
        dominant = real_order[0:num_dominant_seqs]
        other_ = real_order[num_dominant_seqs::]
    else:
        dominant = real_order
        other_ = []

    amounts = []
    relative_amounts = []
    pair_outcomes = []
    pairs = []
    for sequence in order:
        ordered = 0
        reverse = 0
        repeat = 0
        misordered = 0
        non_to_task = 0
        task_to_non = 0
        other = 0
        for index,element in enumerate(sequence[0:-1]):
            pair = [element,sequence[index+1]]
            outcome = (logic_machine_for_pair_catagorisation(pair,dominant,other_))
            if outcome == 'ordered':
                ordered +=1
            elif outcome == 'reverse':
                reverse +=1
            elif outcome == 'repeat':
                repeat +=1
            elif outcome == 'misordered':
                misordered +=1
            elif outcome == 'task to other':
                task_to_non +=1
            elif outcome == 'other to task':
                non_to_task +=1
            elif outcome == 'other':
                other +=1
            pairs += [pair]
            pair_outcomes += [[outcome]]
        pairs += [[None]]
        pair_outcomes += [[None]]

        relative_amounts += [list(np.array([ordered,reverse,repeat,misordered,non_to_task,task_to_non,other])/(len(sequence)-1))]
        amounts += [[ordered,reverse,repeat,misordered,non_to_task,task_to_non,other]]
        
    return relative_amounts, amounts,pair_outcomes,pairs

def logic_machine_for_pair_catagorisation(pair,dominant,other):
    # if first one in dominant check for ordering:
    if pair[0] in dominant and pair[-1] in dominant:
        if pair_in_sequence(pair,dominant):
            return('ordered')
        elif pair_in_sequence(pair,dominant[::-1]):
            return('reverse')
        elif pair[-1] == pair[0]:
            return('repeat')
        elif pair[-1] in dominant:
            return('misordered') 
    # if its not these  options then check if it could be in the extra task seqs
    elif pair[0] in  (dominant + other) and pair[-1] in  (dominant + other):
        for item in other:
            if pair[0] in  (dominant + [item]):
                if pair_in_sequence(pair,(dominant + [item])):
                    return('ordered')
                elif pair_in_sequence(pair,(dominant + [item])[::-1]):
                    return('reverse')
                elif pair[-1] == pair[0]:
                    return('repeat')
                elif pair[-1] in (dominant + [item]):
                    return('misordered')  
        # if not this then check if both are in the extra seqs (and are not a repeat):
        if pair[0] in other and pair[-1] in other:
            if not pair[-1] == pair[0]: 
                return('ordered')
    else:
        # if item 1 is in but item 2 isnt then task to other 
        if pair[0] in  (dominant + other):
            if not pair[-1] in  (dominant + other):
                return('task to other')
        # if item 2 is in but item 1 isnt then other to task 
        elif not pair[0] in  (dominant + other):
            if pair[-1] in  (dominant + other):
                return('other to task')
            else:
                return('other')
    return print('ERROR!')

def pair_in_sequence(pair, sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            return True
        # because its ciruclar:
        elif sequence[-1] == pair[0] and sequence[0] == pair[1]:
            return True
    return False

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


def plot_event_transitions(event_transitions,axes,title):
    transitions = list(event_transitions.keys())
    counts = list(event_transitions.values())

    # Create a bar chart
    axes.bar(range(len(transitions)), counts,alpha = 0.5)

    # Set x-axis labels
    labels = [str(transition) for transition in transitions]
    axes.set_xticks(range(len(transitions)), labels, rotation=90)

    # Set y-axis label
    axes.set_ylabel('Normalized Occurrences %')

    # Add title
    axes.set_title('Event Transitions' + title)

def count_event_transitions(event_list):
    
    ### change code so that it sets each dictionary with these pairs a sa starting point! 
    from itertools import product
    numbers = range(1, 7)  # Numbers 1 to 6a
    all_possible_pairs = list(product(numbers, repeat=2))

    transitions = {key: 0 for key in all_possible_pairs}  # Create an empty dictionary with the defined keys
    
    total_transitions = len(event_list) - 1
    
    for i in range(total_transitions):
        current_event = event_list[i]
        next_event = event_list[i + 1]
        transition = (current_event, next_event)
        if transition in transitions:
            transitions[transition] += 1
        else:
            transitions[transition] = 1
    
    normalized_transitions = {}
    for transition, count in transitions.items():
        normalized_count = count / total_transitions * 100
        normalized_transitions[transition] = normalized_count
    
    return normalized_transitions


def normalize_counts_to_percentages(events_dict):
    total_count = sum(events_dict.values())
    normalized_dict = {}

    for event, count in events_dict.items():
        percentage = (count / total_count) * 100
        normalized_dict[event] = percentage

    return normalized_dict

def SaveFig_noclose(file_name,figure_dir):
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir + file_name, bbox_inches='tight')


def closest_points(target, points, threshold):
    import math
    closest = []
    indicies = []
    for index,point in enumerate(points):
        distance = math.dist(target,point)
        if distance <= threshold:
            closest.append(point)
            indicies.append(index)
    return closest,indicies

def find_next_val(index,threshold_breaks,frame_filter,port_type):
    p2_indicies = np.where(threshold_breaks == port_type)[0]
    try:
        p2_min_val = min(i for i in p2_indicies if i > index)
        distance = p2_min_val - index
    except:
        distance = 9999999
    if distance<frame_filter:
        return p2_min_val
    else:
        return -1

def find_p5_segments(threshold_breaks, frame_filter, find_next_val):
    start_ind = []
    end_ind = []
    P5_in_ind = []  # kept for parity, though unused in original code

    index = 0
    n = len(threshold_breaks)

    while index < n - 1:
        break_ = threshold_breaks[index]

        if break_ == 5 and threshold_breaks[index + 1] != 5:
            p2_ind = find_next_val(index, threshold_breaks, frame_filter, 2)
            if p2_ind == -1:
                index += 1
                continue

            p3_ind = find_next_val(p2_ind, threshold_breaks, frame_filter, 3)
            if p3_ind == -1:
                index += 1
                continue

            p4_ind = find_next_val(p3_ind, threshold_breaks, frame_filter, 4)
            if p4_ind == -1:
                index += 1
                continue

            p5_ind = find_next_val(p4_ind, threshold_breaks, frame_filter, 5)
            if p5_ind == -1:
                index += 1
                continue

            p5_out_ind = find_next_val(p5_ind, threshold_breaks, frame_filter, 0)
            if p5_out_ind == -1:
                index += 1
                continue

            start_ind.append(index)
            end_ind.append(p5_out_ind)

            # jump index forward
            index = p5_out_ind - 1
        else:
            index += 1

    return start_ind, end_ind

def find_closest_point(target, points):
    import math
    min_distance = float('inf')
    closest_point = None
    closest_index = None
    for index, point in enumerate(points):
        distance = math.dist(target,point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
            closest_index = index
    return closest_point,closest_index

def find_closest_points(traject_coords,port_centroids):
    
    split_trajects_port_points = []
    split_trajects_port_indicies = []

    closest_points = []
    closest_inds = []
    for ind_,centroids in enumerate(port_centroids[1::]):
        # skip 0 ^see above and skip port 3: 
        if not centroids == port_centroids[2]:
            if not centroids == port_centroids[-1]:
                if ind_ ==  0:
                    # find closest to port 2
                    closest_point, closest_index = find_closest_point(centroids,traject_coords)
                    closest_points += [closest_point]
                    closest_inds += [closest_index]
                else:
                    #find closest to port 4 (from after port 2 onwards)
                    closest_point, closest_index = find_closest_point(centroids,traject_coords[closest_inds[-1]::])
                    closest_points += [closest_point]
                    closest_inds += [closest_index + closest_inds[-1]]
            else:
                # if port 5 search from port 4 onwards
                closest_point, closest_index = find_closest_point(centroids,traject_coords[closest_inds[-1]::])
                closest_points += [closest_point]
                closest_inds += [closest_index + closest_inds[-1]]

    ###make it so that point closest to port 1 can only be between port 5 and 2! 
    closest_point_1, closest_index_1 = find_closest_point(port_centroids[0],traject_coords[0:closest_inds[0]])

    ###make it so that point closest to port 3 can only be between port 2 and 4! 
    closest_point_3, closest_index_3 = find_closest_point(port_centroids[2],traject_coords[closest_inds[0]:closest_inds[1]])
    # this finds relative ind so I need to add on the offst from the start of the full traject
    closest_index_3 = closest_index_3 + closest_inds[0]

    split_trajects_port_points += [[closest_point_1] + [closest_points[0]] + [closest_point_3] + closest_points[1::]]
    split_trajects_port_indicies += [[closest_index_1] + [closest_inds[0]] + [closest_index_3] + closest_inds[1::]]
    
    return split_trajects_port_points, split_trajects_port_indicies


def interpolate_to_longest_and_find_average_curve(curves):
    
    

    # Find the length of the longest curve
    max_length = max([len(curve) for curve in curves])

    # Interpolate each curve to the length of the longest curve
    interpolated_curves = []
    for curve in curves:
        if len(curve) > 0:
            x = [point[0] for point in curve]
            y = [point[1] for point in curve]

            # find lots of points on the piecewise linear curve defined by x and y
            M = max_length
            t = np.linspace(0, len(x), M)
            x_interp = np.interp(t, np.arange(len(x)), x)
            y_interp = np.interp(t, np.arange(len(y)), y)

            interpolated_curves.append([[x, y] for x, y in zip(x_interp, y_interp)])

    # # Average the x and y coordinates of all the interpolated curves
    average_curve = []
    for i in range(max_length):
        x_sum = 0
        y_sum = 0
        for curve in interpolated_curves:
            x_sum += curve[i][0]
            y_sum += curve[i][1]
        average_curve.append([x_sum / len(interpolated_curves), y_sum / len(interpolated_curves)])

    return average_curve


def total_length_of_curve(curve):
    x = [point[0] for point in curve]
    y = [point[1] for point in curve]
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(dists)


def closest_point(line1, line2):
    tree = KDTree(line2)
    dist, index = tree.query(line1)
    return index, dist


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

# Function to find corresponding number in another column
def find_corresponding(nums,df_dict):
    return [df_dict[num] for num in nums]

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

def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def load_H5_bodypart(tracking_path,video_type, tracking_point):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file or video_type.upper() in file:
            if 'BACK' in file or 'task' in file:
                if not 'PORT' in file:
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
        if video_type in file or video_type.upper() in file:
            if 'PORT' in file or 'port' in file:
                back_ports_file = pd.read_hdf(tracking_path + file)

    ## same for the ports:
    scorer = back_ports_file.columns.tolist()[0][0]
        
    try:
        port1 =back_ports_file[scorer]['port2']
        port2 =back_ports_file[scorer]['port1']
        port3 =back_ports_file[scorer]['port6']
        port4 =back_ports_file[scorer]['port3']
        port5 =back_ports_file[scorer]['port7']
    except:
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


def convolve_movmean(y,N):
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth

# Function to find the closest index using np.searchsorted
def find_closest_indices(timestamps, target_times):
    indices = np.searchsorted(timestamps, target_times)
    indices = np.clip(indices, 1, len(timestamps) - 1)  # Ensure indices are within bounds
    
    # Compare target_time with its neighbors to find the closest
    left_indices = indices - 1
    right_indices = indices
    left_diffs = np.abs(timestamps[left_indices] - target_times)
    right_diffs = np.abs(timestamps[right_indices] - target_times)
    
    return np.where(left_diffs <= right_diffs, left_indices, right_indices)