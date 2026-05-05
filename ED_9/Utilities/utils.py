
import os 
from pathlib import Path
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy as np
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import scipy
import re
from skbio.stats.distance import permanova, DistanceMatrix
from skbio import DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

def extract_data_warp(df,mouse_standard_label,standard_insertion_mislabeled, standard_insertion):

    #extract data properly into dfs: 
    number = []
    proportion = []
    mislabelled_proportion =[]
    mir = []
    current_type = 'warp' 
    for index,type_ in enumerate(df.type):
        if current_type in type_:
            if not np.isnan(df.proportion_found[index]):
                proportion += [(df.proportion_found[index]*100)]
                mislabelled_proportion += [(df.proportion_mislabeled[index]*100)]
                mir += [df.mouse_implant_recording[index]]
                try:
                    number += [float(re.findall(r'\d+\.\d+', type_)[0])]
                    if float(re.findall(r'\d+\.\d+', type_)[0]) == 0.1:
                        print(df.mouse_implant_recording[index])
                except:
                    number += [int(re.findall(r'\d+', type_)[0])]
                    
    number += [1]*len(standard_insertion)
    proportion += list(standard_insertion)
    mislabelled_proportion += list(standard_insertion_mislabeled)
    mir += list(mouse_standard_label)

    # sort these
    proportion = np.array(proportion)[np.argsort(number)]
    mislabelled_proportion = np.array(mislabelled_proportion)[np.argsort(number)]
    total_found = proportion 
    mir = np.array(mir)[np.argsort(number)]  
    number = np.array(number)[np.argsort(number)]    

    out_df = pd.DataFrame({'mir':mir,'number':number,'norm_total_found':total_found,'norm_mislabelled_proportion':mislabelled_proportion})
    
    return(out_df)


def plot_method_warp(
    ax,
    ax2,
    df,
    x,
    *,
    color,
    label_prefix,
    group_col='number',
    found_col='norm_total_found',
    mislab_col='norm_mislabelled_proportion',
):
    # ---- found ----
    found_means, found_stds = compute_group_stats(df, found_col, group_col)


    plot_with_std(
        ax,
        x,
        found_means,
        found_stds,
        color=color,
        label=f"{label_prefix} event found",
        marker='o-',
        alpha_fill=0.4,
    )

    # ---- mislabelled ----
    mis_means, mis_stds = compute_group_stats(df, mislab_col, group_col)

    plot_with_std(
        ax2,
        x,
        mis_means,
        mis_stds,
        color=color,
        label=f"{label_prefix} mislabelled",
        marker='P--',
        alpha_fill=0.2,
        alpha_line=0.4,
    )

def extract_stats_inputs(df_dict,standard_insertion,standard_insertion_mislabeled,decoder_standard_insertion,decoder_standard_insertion_mislabeled,decoder2_standard_insertion,decoder2_standard_insertion_mislabeled):
    """
    Extract found + mislabelled data for stats
    """
    found = {}
    mislabeled = {}

    found["ppseq"], mislabeled["ppseq"] = extract_data_for_stats(
        df_dict["ppseq"],
        standard_insertion,
        standard_insertion_mislabeled,
    )

    found["decoder"], mislabeled["decoder"] = extract_data_for_stats(
        df_dict["decoder"],
        decoder_standard_insertion,
        decoder_standard_insertion_mislabeled,
    )

    found["decoder2"], mislabeled["decoder2"] = extract_data_for_stats(
        df_dict["decoder2"],
        decoder2_standard_insertion,
        decoder2_standard_insertion_mislabeled,
    )

    return found, mislabeled


# =====================================================
# Statistics helpers
# =====================================================
def check_normality(datasets, alpha=0.05):
    """
    Run Shapiro tests and warn if non-normal
    """
    for i, group in enumerate(datasets):
        for item in group:
            p = shapiro(item)[1]
            if p < alpha:
                print(f"Group {i}: non-normal detected (p={p:.4g}) → use non-parametric")
                break


def run_triplewise_permanova(data_groups, label):
    """
    Run PERMANOVA + feature-wise triplewise comparisons
    """
    print(f"\n{'='*30} {label.upper()} {'='*30}")

    dm, grouping, stats_df = perform_permanova(*data_groups)

    features = stats_df.iloc[:, 1:].values
    feature_results = triplewise_permanova_by_feature(
        features,
        np.array(grouping),
    )

    print("Feature-wise Triplewise Results:")
    for feature_index, g1, g2, g3, result in feature_results:
        p = result["p-value"]
        if p < 0.05:
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            print("^^^^^^^^^^^^^^^^^^^^^^")
            print(
                f"Feature {feature_index} / {features.shape[1]-1}: "
                f"Comparison: {g1}, {g2}, {g3}"
            )
            print(f"P-value (corrected): {p:.4g} {stars}")
            print(result)
            print("______")



def replace_nans(list_of_lists):
    result = []
    for lst in list_of_lists:
        # Convert list to numpy array for ease of handling NaNs
        array = np.array(lst, dtype=np.float64)
        # Calculate mean of non-NaN values
        mean_value = np.nanmean(array)
        # Replace NaNs with the mean value
        array[np.isnan(array)] = mean_value
        # Convert back to list and append to result
        result.append(array.tolist())
    return result

def perform_permanova(data1,data2,data3):

    data1 = replace_nans(data1)
    data2 = replace_nans(data2)
    data3 = replace_nans(data3)

    stats_df = pd.DataFrame()

    longest_length1 = max(len(lst) for lst in data1)
    longest_length2 = max(len(lst) for lst in data2)
    longest_length3 = max(len(lst) for lst in data3)

    stats_df['groups'] = ['ppseq']*longest_length1  + ['decod1']*longest_length2 + ['decod2']*longest_length3



    for index,item in enumerate(data1):
        while len(data1[index]) < longest_length1:
            data1[index] += [np.mean(data1[index])]
        while len(data2[index]) < longest_length2:
            data2[index] += [np.mean(data2[index])]
        while len(data3[index]) < longest_length3:
            data3[index] += [np.mean(data3[index])]
        stats_df[str(index)] = data1[index] + data2[index] + data3[index]


    # Calculate the Euclidean distance matrix
    values = stats_df[list(stats_df)[1::]].values
    grouping = stats_df['groups'].values

    pairwise_distances = pdist(values, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    dm = DistanceMatrix(distance_matrix)

    # Perform PERMANOVA
    results = permanova(dm, grouping, permutations=10000)
    print('-----------------')
    print(results)
    return dm, grouping,stats_df

def triplewise_permanova_by_feature(data, group_labels, method='bonferroni'):
    unique_groups = np.unique(group_labels)
    triplet_combinations = list(combinations(unique_groups, 3))
    feature_results = []
    feature_p_values = []

    num_features = data.shape[1]

    for feature_index in range(num_features):
        feature_data = data[:, feature_index]

        for group1, group2, group3 in triplet_combinations:
            mask = np.isin(group_labels, [group1, group2, group3])
            triplewise_feature_data = feature_data[mask]
            triplewise_group_labels = group_labels[mask]

            # Compute the distance matrix for the feature
            pairwise_distance_matrix = squareform(pdist(triplewise_feature_data[:, np.newaxis], metric='euclidean'))

            # Ensure the array is contiguous
            pairwise_distance_matrix = np.ascontiguousarray(pairwise_distance_matrix)

            # Create a DistanceMatrix object
            ids = np.arange(len(triplewise_group_labels))
            pairwise_distance_matrix = DistanceMatrix(pairwise_distance_matrix, ids)

            result = permanova(pairwise_distance_matrix, triplewise_group_labels, permutations=10000)
            feature_results.append((feature_index, group1, group2, group3, result))
            feature_p_values.append(result['p-value'])

    # Apply Bonferroni correction
    corrected_p_values = multipletests(feature_p_values, method=method)[1]

    # Update results with corrected p-values
    for i in range(len(feature_results)):
        feature_results[i][4]['p-value'] = corrected_p_values[i]

    return feature_results
    


def extract_data_for_stats(df_func,standard_insertion,standard_insertion_mislabeled):
    data1 = []
    data4 = []
    counter = 0
    for number, group in  df_func.groupby('number'):
        if counter == 0:
            if not number == 0:
                # if no standard add it back in
                data1 += [list(standard_insertion)]
                data4 += [list(standard_insertion_mislabeled)]
        data1 += [list(group.norm_total_found)]
        data4 += [list(group.norm_mislabelled_proportion)]
        counter +=1
    return data1, data4



def extract_data(df,type_label,mouse_standard_label,standard_insertion):

    #extract data properly into dfs: 
    number = []
    proportion = []
    mislabelled_proportion =[]
    mir = []
    current_type = type_label 
    for index,type_ in enumerate(df.type):    
        if current_type in type_:
            if not np.isnan(df.proportion_found[index]):
                number += [int(re.findall(r'\d+', type_)[0])]
                standard_found = standard_insertion[np.where(mouse_standard_label== df.mouse_implant_recording[index])[0][0]]
                proportion += [(df.proportion_found[index]*100)]
                mislabelled_proportion += [(df.proportion_mislabeled[index]*100)]
                mir += [df.mouse_implant_recording[index]]


    # sort these
    proportion = np.array(proportion)[np.argsort(number)]
    mislabelled_proportion = np.array(mislabelled_proportion)[np.argsort(number)]
    total_found = proportion 
    mir = np.array(mir)[np.argsort(number)]  
    number = np.array(number)[np.argsort(number)]    

    out_df = pd.DataFrame({'mir':mir,'number':number,'norm_total_found':total_found,'norm_mislabelled_proportion':mislabelled_proportion})
    
    return(out_df)

def calc_group_values(data,number):
    group_values = {}
    for value, group in zip(data, number):
        if group in group_values:
            group_values[group].append(value)
        else:
            group_values[group] = [value]
    group_means = {}
    group_std ={}
    for group, values in group_values.items():
        group_means[group] = np.mean(values)
        group_std[group] = scipy.stats.sem(values)
    print(group_means)
    return group_values,group_means,group_std


def compute_group_stats(df, value_col, group_col):
    """
    Wrapper around calc_group_values for readability
    """
    _, means, stds = calc_group_values(
        df[value_col].values,
        df[group_col].values
    )
    return means, stds


def plot_with_std(
    ax,
    x,
    mean,
    std,
    *,
    color,
    label,
    marker='o-',
    alpha_line=0.6,
    alpha_fill=0.4,
    markersize=10,
):
    mean_arr = np.array(list(mean.values()))
    std_arr = np.array(list(std.values()))

    upper = mean_arr + std_arr
    lower = mean_arr - std_arr

    ax.plot(
        x,
        mean.values(),
        marker,
        color=color,
        alpha=alpha_line,
        label=label,
        markeredgewidth=0,
        markersize=markersize,
    )

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=alpha_fill,
        facecolor=color,
        edgecolor='None',
        linewidth=1,
        linestyle='dashdot',
        antialiased=True,
    )


def plot_method(
    ax,
    ax2,
    df,
    *,
    color,
    label_prefix,
    group_col='number',
    found_col='norm_total_found',
    mislab_col='norm_mislabelled_proportion',
):
    # ---- found ----
    found_means, found_stds = compute_group_stats(df, found_col, group_col)
    x = list(found_means.keys())

    plot_with_std(
        ax,
        x,
        found_means,
        found_stds,
        color=color,
        label=f"{label_prefix} event found",
        marker='o-',
        alpha_fill=0.4,
    )

    # ---- mislabelled ----
    mis_means, mis_stds = compute_group_stats(df, mislab_col, group_col)

    plot_with_std(
        ax2,
        x,
        mis_means,
        mis_stds,
        color=color,
        label=f"{label_prefix} mislabelled",
        marker='P--',
        alpha_fill=0.2,
        alpha_line=0.4,
    )
    
    
def load_h5(filename):
    def decode_and_convert(x):
        if isinstance(x, bytes):
            x = x.decode('utf-8')
        if isinstance(x, str):
            try:
                return float(x)
            except ValueError:
                return x
        return x
    def load_item(obj):
        # -------------------------------------------------------------
        # GROUP HANDLING
        # -------------------------------------------------------------
        if isinstance(obj, h5py.Group):
            # --- DataFrame detection ---
            if "data" in obj and "columns" in obj.attrs:
                cols = [c.decode("utf-8") for c in obj.attrs["columns"]]
                raw = obj["data"][()]
                conv = np.array(
                    [[decode_and_convert(cell) for cell in row] for row in raw]
                )
                return pd.DataFrame(conv, columns=cols)

            # --- List group detection ---
            keys = list(obj.keys())
            if keys and all(k.isdigit() for k in keys):
                return [load_item(obj[k]) for k in sorted(keys, key=int)]

            # --- Regular dict ---
            return {k: load_item(obj[k]) for k in keys}
        # -------------------------------------------------------------
        # DATASET HANDLING
        # -------------------------------------------------------------
        elif isinstance(obj, h5py.Dataset):
            data = obj[()]
            # --- Scalar bytes ---
            if isinstance(data, bytes):
                return decode_and_convert(data)
            # --- Scalar string ---
            if isinstance(data, str):
                return decode_and_convert(data)
            # --- Scalar numeric ---
            if not isinstance(data, np.ndarray):
                return data
            # --- Array of bytes or objects ---
            if data.dtype.kind in {"S", "O"}:
                flat = np.array([decode_and_convert(x) for x in data.flat])
                if flat.ndim == 1:
                    return flat.tolist()
                return flat.reshape(data.shape)
            # --- Numeric array ---
            return data
        return obj
    with h5py.File(filename, "r") as f:
        return {k: load_item(f[k]) for k in f.keys()}
    


