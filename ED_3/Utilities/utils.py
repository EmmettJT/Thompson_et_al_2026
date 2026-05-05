import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy
from scipy import stats
from scipy.stats import shapiro, normaltest
import json 
import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

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
    

def get_percentage_points(resampled_curve, percentages):
    """
    Get the points at specific percentage locations along the resampled curve.
    
    :param resampled_curve: np.array of shape (N, 2), evenly spaced curve points
    :param percentages: List of percentages (0-100) where points should be extracted
    :return: List of (x, y) points corresponding to the given percentages
    """
    num_points = len(resampled_curve)
    indices = [int(p / 100 * (num_points - 1)) for p in percentages]  # Convert percentage to index
    return resampled_curve[indices]

def plot_percentage_interval(resampled_curve, percentages,port_centroids,ax,sequence_name,colour_,radius_used):
    """
    Plot the full resampled curve and highlight the region between two percentage points.
    
    :param resampled_curve: np.array of shape (N, 2), evenly spaced curve points
    :param percentages: List [start%, end%] defining the highlighted region
    """
    num_points = len(resampled_curve)
    start_idx = int(percentages[0] / 100 * (num_points - 1))
    end_idx = int(percentages[1] / 100 * (num_points - 1))

    # Handle case where interval wraps around (e.g., 97% to 14%)
    if start_idx <= end_idx:
        highlight_indices = range(start_idx, end_idx + 1)
    else:
        highlight_indices = list(range(start_idx, num_points)) + list(range(0, end_idx + 1))

    # Extract x, y coordinates
    x_vals, y_vals = resampled_curve[:, 0], resampled_curve[:, 1]

    # Highlight section
    highlighted_x = x_vals[list(highlight_indices)]
    highlighted_y = y_vals[list(highlight_indices)]
    ax.plot(highlighted_x, highlighted_y, color=colour_, linewidth=5, label="Highlighted Region")

    for index,port_centroid in enumerate(port_centroids):
        circle1 = plt.Circle(port_centroid, radius_used, color='grey', alpha = 0.2)
        ax.add_patch(circle1)

        
    ax.invert_yaxis()

def plot_sleep_awake_spike_position(ax,x,y,color_):
    # Create a pandas DataFrame
    df = pd.DataFrame({'x': x, 'y': y})

    # Define bin edges for y-values
    bin_edges = np.linspace(0, 1, num=15)  # 5 bins from 0 to 1

    # Bin the data based on y-values
    df['bin'] = pd.cut(df['y'], bins=bin_edges)

    # Calculate the middle points of the bins
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the mean and standard deviation of each bin
    bin_means = df.groupby('bin')['x'].mean()
    bin_std = df.groupby('bin')['x'].std()

    # Plot

    ax.plot(bin_midpoints,bin_means,'o-', c = color_, markersize = 8)
    # plt.plot(bin_midpoints, bin_means, width=0.1, align='center', label='Mean')

    # Fill between mean line +- standard deviation
    plt.fill_between(bin_midpoints, bin_means - bin_std, bin_means + bin_std, color=color_, alpha=0.3)
    
    return bin_means.values,bin_midpoints

def test_univariate_normality(x, y, alpha=0.05):
    """
    Performs Shapiro-Wilk and D'Agostino's tests on each axis.
    Prints p-values and a simple verdict.
    """
    for name, data in (('x', x), ('y', y)):
        W, p_sw = shapiro(data)
        K2, p_k2 = normaltest(data)
        print(f"--- {name}-axis ---")
        print(f" Shapiro–Wilk:    W={W:.3f}, p={p_sw:.3f} -> {'non-normal' if p_sw<alpha else 'normal'}")
        print(f" D’Agostino K²: K²={K2:.3f}, p={p_k2:.3f} -> {'non-normal' if p_k2<alpha else 'normal'}\n")

    

def return_binned_neuron_awake_sleep_rel(awake_dat,sleep_dat):
    awake_sleep_relationship = {}
    for index_, item, in enumerate(awake_dat):
        for index, awake_mouse_data in enumerate(item):

            awake = awake_mouse_data
            sleep = np.array(sleep_dat[index_][index])

            min_neurons_filter = 0
            bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]    

            for bin_ in bins:
                mask = np.round(awake,1) == bin_
                if len(sleep[mask]) > min_neurons_filter:
                    if bin_ in awake_sleep_relationship:
                        awake_sleep_relationship[bin_] += [np.mean(sleep[mask])]
                    else:
                        awake_sleep_relationship[bin_] = [np.mean(sleep[mask])]

    return awake_sleep_relationship


def plot_awake_sleep_relationship(awake_sleep_relationship,color_,ax):

    keys = []
    e_means = []
    sem = []
    for key in awake_sleep_relationship:
        if len(awake_sleep_relationship[key]) > 0:
            keys += [key]
    #         ax.plot([key]* len(e_awake_sleep_relationship[key]),e_awake_sleep_relationship[key],'o', color = 'red')

            a_s_relationship = [x for x in awake_sleep_relationship[key] if not np.isnan(x)]

            e_means += [np.mean(a_s_relationship)]
            sem += [scipy.stats.tstd(a_s_relationship)]
    ax.plot(sorted(keys),np.array(e_means)[np.argsort(keys)],'o--', c = color_, alpha = 1, markeredgewidth = 0, markersize = 10)

    upper = np.array(e_means)[np.argsort(keys)] + sem
    lower = np.array(e_means)[np.argsort(keys)] - sem
    ax.fill_between(sorted(keys),(lower),(upper),
        alpha=0.2, edgecolor='None', facecolor=color_,
        linewidth=1, linestyle='dashdot', antialiased=True)

    x = np.linspace(0, 1, 10)  # Example x values
    y = x  # Since x equals y, y values are the same as x values
    plt.plot(x,y,'--')

    ax.set_ylim(0,0.8)
    ax.set_ylabel('replay proportion involvement')
    ax.set_xlabel('awake proportion involvement')

    #################################################

    from scipy.optimize import curve_fit
    from scipy.stats import t

    # Extract the bins and corresponding values
    bins = list(awake_sleep_relationship.keys())
    
    bin_centers = np.array(bins)[np.argsort(bins)]
    values = np.array([np.nanmean(awake_sleep_relationship[bin]) for bin in bins])[np.argsort(bins)]

    # Define the exponential function
    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    # Fit the model to the data
    popt, pcov = curve_fit(exponential_model, bin_centers, values)

    # Extract the fitting parameters
    a, b = popt

    # Calculate the fitted values
    fitted_values = exponential_model(bin_centers, *popt)

    # Calculate R-squared
    residuals = values - fitted_values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((values - np.mean(values))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Degrees of freedom
    dof = max(0, len(bin_centers) - len(popt))

    # Calculate standard errors of the parameters
    perr = np.sqrt(np.diag(pcov))

    # Calculate t-values for the parameters
    t_values = popt / perr

    # Calculate p-values for the parameters
    p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]

    print(f'Fitted parameters: a = {a}, b = {b}')
    print(f'R-squared: {r_squared}')
    print(f'p-values: {p_values}')

    # Plot the original data and the fitted curve
    # plt.scatter(bin_centers, values, label='Data')
    # plt.plot(bin_centers, fitted_values, label='Fitted curve', color='red', alpha = 0.4)
    plt.xlabel('Bin Center')
    plt.ylabel('Value')
