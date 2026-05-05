import os
import glob
import pandas as pd
import numpy as np
import ast
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns; 
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy
import pickle 
import math
from scipy.spatial import KDTree
import random
import statsmodels
import scipy.stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from statistics import mean
from scipy import stats
import h5py
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

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_sd


def mean_learning_curve(AA_TrainingLevels):

    TrialbyTrial_Pscores = conactinate_nth_items(AA_TrainingLevels)
    MeanLearningCurve = []
    for i,item in enumerate(TrialbyTrial_Pscores):
        MeanLearningCurve = MeanLearningCurve + [np.mean(item)]

    return MeanLearningCurve

def standard_dev_across_trials(AA_TrainingLevels):

    TrialbyTrial_tlevels = conactinate_nth_items(AA_TrainingLevels)

    standard_dev_bytrial = []
    for trial in TrialbyTrial_tlevels:
        standard_dev_bytrial = standard_dev_bytrial + [np.std(trial)]

    return standard_dev_bytrial

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



def CreateSequences_Time(Transition_types,Transition_times,transition_reference_time,Transition_filter_time):
    # reoder transitions into time releveant sequences  
    seq_index = 0
    TimeFiltered_ids = [[]]
    TimeFiltered_times = [[]]
    Reference_times = [[]]

    for ind, transit in enumerate (Transition_types):
        if Transition_times[ind]:
            if Transition_times[ind] < Transition_filter_time and Transition_times[ind] > 0.03: # if less than filter time and more than lower bound filter time:
                TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
                TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]
                Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]


            else:
                if TimeFiltered_ids[seq_index]: # if not empty 
                    seq_index = seq_index + 1
                    TimeFiltered_ids = TimeFiltered_ids + [[]]
                    TimeFiltered_times = TimeFiltered_times +[[]] 
                    Reference_times = Reference_times + [[]]
                    TimeFiltered_ids[seq_index] = TimeFiltered_ids[seq_index] + [transit]
                    TimeFiltered_times[seq_index] = TimeFiltered_times[seq_index] + [Transition_times[ind]]   
                    Reference_times[seq_index] = Reference_times[seq_index] + [transition_reference_time[ind]]
    return TimeFiltered_ids,TimeFiltered_times,Reference_times


def determime_heatmapdata(var,port1,port2,port3,port4):
    TransitionTypesIndex = np.array([11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48,51,52,53,54,55,56,57,58,61,62,63,64,65,66,67,68,71,72,73,74,75,76,77,78,81,82,83,84,85,86,87,88])
    port1_transits = []
    for i in range(((port1*8)-8),((port1*8)-8)+8):
        port1_transits = port1_transits + [var[i]]

    port2_transits = []
    for i in range(((port2*8)-8),((port2*8)-8)+8):
        port2_transits = port2_transits + [var[i]]

    port3_transits = []
    for i in range(((port3*8)-8),((port3*8)-8)+8):
        port3_transits = port3_transits + [var[i]]

    port4_transits = []
    for i in range(((port4*8)-8),((port4*8)-8)+8):
        port4_transits = port4_transits + [var[i]]

    port_transits = [port1_transits] + [port2_transits] + [port3_transits] + [port4_transits]

    return port_transits

def determineTransitionNumber(TimeFiltered_seqs):
    TransitionTypesIndex = np.array([11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48,51,52,53,54,55,56,57,58,61,62,63,64,65,66,67,68,71,72,73,74,75,76,77,78,81,82,83,84,85,86,87,88])
    trajects = []
    for inds, seqs in enumerate(TimeFiltered_seqs):
#         seqs = literal_eval(seqs) # convert back from weird df string conversion thing
        for ind, transits in enumerate(seqs):
#             if not str(transits)[0] == str(transits)[1]:
            trajects = np.append(trajects,transits)
    transition_number = []
    for transit_types in TransitionTypesIndex:
        temp = (np.where(trajects == float(transit_types)))
        transition_number.append(len(temp[0]))
    return transition_number

def reversedata(port_transits,new_order):

    reordered_port_transits = []
    for i in range(1,len(port_transits)+1):
        mylist = port_transits[-i]
        myorder = new_order
        mylist = [mylist[i] for i in myorder]
        newlist = []
        for item in mylist:
            newlist = newlist + [float(item)]
        reordered_port_transits = reordered_port_transits + [newlist]
    #restructure data to swap x and y axis:
    data = [[],[],[],[],[],[],[],[]]
    for ind in range(8):
        for index,item in enumerate(reordered_port_transits):
            data[ind] = data[ind] + [item[len(item)-1-ind]]
    for i in range(8):
        data[i].reverse()
    return data

def port_fitted_poke_times(Fitted_tfiltered_seqs,Fitted_tfiltered_times,port,max_filter):
    PortPokes = [[],[],[],[],[],[],[],[]]
    for index, seq in enumerate(Fitted_tfiltered_seqs):
        seq = literal_eval(seq) 
        if np.size(seq) > 0:
            if int(str(seq[0])[0]) == port: # if sequence starts with port
                if not int(str(seq[0])[1]) == port: #ignore self pokes 
                    current_seq_time = 0
                    for ind,item in enumerate(seq):
                        if ind > max_filter: # ignore long chains of seqs that dont return to the start port as they skew data towards being super long..
                            break
                        c_port = int(str(item)[-1])-1
                        PortPokes[c_port] = PortPokes[c_port] +[current_seq_time + literal_eval(Fitted_tfiltered_times[index])[ind]]
                        current_seq_time = current_seq_time + literal_eval(Fitted_tfiltered_times[index])[ind]
    return PortPokes

def create_plotting_df(portpokes,new_order):
    concatinated = []
    ids = []
    for index, port in enumerate(new_order):
        concatinated = concatinated + portpokes[port] 
        if index < 5:
            ids = ids +  (len(portpokes[port]) * [index])
        else:
            ids = ids +  (len(portpokes[port]) * [5])
            
    df = pd.DataFrame({'index' : ids, 
                       'Time':concatinated})
    return(df)

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


def parse_training_levels(training_levels):
    t_levels = []
    for row in training_levels:
        row = row.replace('nan', 'None')
        row = '[' + ', '.join(convert_float_string(x) for x in row.strip('[]').split(', ')) + ']'
        t_levels.append(literal_eval(row))
    return t_levels

def calculate_mean_std(t_levels, mask):
    trial_scores = conactinate_nth_items(np.array(t_levels)[mask])
    mean_curve = [np.mean(item) for item in trial_scores]
    std_curve = [np.std(item) for item in trial_scores]
    return mean_curve, std_curve

def fill_between_mean_std(ax, mean_curve, std_curve, color,xlim):
    upper = np.array(mean_curve[:xlim]) + np.array(std_curve[:xlim])
    lower = np.array(mean_curve[:xlim]) - np.array(std_curve[:xlim])
    upper[upper > 50] = 50  # Ceiling effect cutoff
    ax.fill_between(range(len(upper)), lower, upper, alpha=0.2, edgecolor='None', facecolor=color, linewidth=1, linestyle='dashdot', antialiased=True)
    

def subset_of_restructured_df(restructured_levels_per_trial_df, column_selection, selection):
    AP5_restructured_levels_per_trial_df  = restructured_levels_per_trial_df.copy()
    AP5_restructured_levels_per_trial_df = AP5_restructured_levels_per_trial_df.loc[AP5_restructured_levels_per_trial_df[column_selection] == selection]
    AP5_restructured_levels_per_trial_df = AP5_restructured_levels_per_trial_df.reset_index(drop=True)
    return AP5_restructured_levels_per_trial_df

def make_example_infusion_plot(all_TrainingLevels):
    
    trials_per_session= []
    for i in range(len(all_TrainingLevels)):
        trials_per_session = trials_per_session + [len(all_TrainingLevels[i])]
    cum_trials_per_session = np.cumsum(trials_per_session)

    #manually define sessions when infusions were done: 
    AP5_sessions = [5, 10]
    Saline_sessions = [8, 12]

    TrainLevelsAll = sum(all_TrainingLevels, [])
    fnt = 24
    #if CurrentAnimal == 'EJT185':
    fig = plt.figure(figsize=(18, 10))
    ax = plt.subplot2grid((5, 3), (1, 0), rowspan =4,colspan =3)   
    ax.set_xlabel('Trials', fontsize = fnt, color = "black")
    ax.plot(TrainLevelsAll,linewidth = 1,color = 'black')
    ax.set_ylabel('Training level',fontsize =fnt, color = 'black')
    ax.tick_params(axis='x', labelsize=fnt)
    ax.tick_params(axis='y', labelsize=fnt)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(True)
    ax.spines.bottom.set_visible(True)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    #ax.set_title(CurrentAnimal)
    ymax = max(TrainLevelsAll)
    ymin = min(TrainLevelsAll)

    mylightorange = "#BF6D73"# "#fff6f2"#"#fff3ea" #"#f7e3d8" #"#e8ad8b" ##f69d6b" #"#fdbf6f"
    mydarkorange =  "#BF6D73"#"#ffdac4" #"#f9d6c0" #"#f7e3d8" #'#fcd2b8'#"#efc8b2" #"#dd8452" #"#ff7f00" #

    mylightblue = "#364D9C"#"#e4f0f7"
    mydarkblue =  "#364D9C" #"#2b7bba" #"#0000FF" #


    ax.set_xlim(1, 3000)
    ax.set_ylim(0.5, 52)

    ax.axvspan(693, 782, color= mylightorange, alpha=0.3)

    ax.axvspan(1551, 1718, color= mylightorange, alpha=0.3) #subtracted 10 from end

    #Saline
    ax.axvspan(1106, 1341, color= mylightblue, alpha=0.3) #subtracted 10 from end
    ax.axvspan(1946, 2247, color= mylightblue, alpha=0.3)

    ax.set_yticks([1, 10, 20, 30, 40 ,50])
    ax.text(2530, 28, "Test session", color = "black", fontsize = fnt)
    ax.text(2460, 22, "Day after infusion", color = "black", fontsize = fnt)
    #ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.text(0.059, 0.272, "Early", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,12), 100, 8,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.059, 0.40, "Middle", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,20.5), 100, 14,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.059, 0.59, "Late", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,35), 100, 15,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.0305, 0.11, 'Naive', color = "gray",  transform=plt.gcf().transFigure, fontsize = fnt)
    ax.text(0.0305, 0.71, 'Expert', color = "gray",  transform=plt.gcf().transFigure, fontsize = fnt)
    arrow_pos = -0.1
    ax.annotate('', xy = (arrow_pos, 0.97), xycoords='axes fraction', xytext=(arrow_pos, 0.04), 
            arrowprops=dict(arrowstyle="->, head_width=0.3, head_length =0.7", color='gray', linewidth = 2))

    for session in Saline_sessions:
        #if CurrentAnimal == 'EJT185':
        infusion_trial = cum_trials_per_session[session - 1]
        infusion_level = all_TrainingLevels[session - 1][-1]
        plt.arrow(infusion_trial, infusion_level + 8, 0, -7.8, length_includes_head = True, width = 13, head_width=40, 
                  head_length = 1.1, color = "#1f78b4") #"#2b7bba")
        ax.text(infusion_trial - 80, infusion_level + 8.5, 'Saline', color = "#1f78b4", fontsize = fnt)


    for infusion_number, session in enumerate(AP5_sessions):
    #         if CurrentAnimal == 'EJT185':
        infusion_trial = cum_trials_per_session[session-1] #i.e. the cumulative number of trials just before the infusion
        infusion_level = all_TrainingLevels[session -1][-1]
        plt.arrow(infusion_trial, infusion_level +8, 0, -7.8, length_includes_head = True, width = 13, head_width=40, 
                  head_length = 1.1, color = "#dd8452")# "darkorange")
        ax.text(infusion_trial - 60, infusion_level + 8.5, 'AP5', color = "#dd8452", fontsize = fnt)
                
    return