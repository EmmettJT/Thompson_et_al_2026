import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# I/O

def load_minimal_data(data_dir):
    data_dir = Path(data_dir)
    lfp = np.load(data_dir / "lfp.npy")
    timestamps = np.load(data_dir / "timestamps.npy")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    return lfp, timestamps, meta


# ---------------------------------------------------------------------------
# Signal processing

def butter_bandpass_filter(data, lowcut, highcut, fs, order_hp=2, order_lp=4, axis=-1):
    nyq = 0.5 * fs
    b_hp, a_hp = butter(order_hp, lowcut / nyq, btype="high")
    filtered_hp = filtfilt(b_hp, a_hp, data, axis=axis)
    b_lp, a_lp = butter(order_lp, highcut / nyq, btype="low")
    return filtfilt(b_lp, a_lp, filtered_hp, axis=axis)


def analytic_amplitude(data, axis=-1):
    return np.abs(hilbert(data, axis=axis))


def smooth_signal(data, sigma_ms=12.5, fs=2500, axis=-1):
    return gaussian_filter1d(data, sigma=sigma_ms * fs / 1000.0, axis=axis, mode="reflect")


def zscore_signal(data, axis=-1):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / std


def preprocess_lfp(lfp, fs=2500):
    lfp_ripple = butter_bandpass_filter(lfp, 150, 250, fs)
    lfp_z = zscore_signal(smooth_signal(analytic_amplitude(lfp_ripple).mean(axis=0), fs=fs))
    lfp_sw = butter_bandpass_filter(lfp, 1, 30, fs)
    return lfp_ripple, lfp_z, lfp_sw


# ---------------------------------------------------------------------------
# Event detection

def _local_minima_boundaries(signal_1d, peak_idx, search_window, baseline=0.0):
    n = len(signal_1d)
    left_bound = max(0, peak_idx - search_window)
    right_bound = min(n - 1, peak_idx + search_window)

    start = peak_idx - 1
    while start > left_bound:
        if signal_1d[start] < signal_1d[start - 1]:
            break
        start -= 1
    if start == left_bound:
        start = peak_idx
        while start > left_bound and signal_1d[start] > baseline:
            start -= 1

    end = peak_idx + 1
    while end < right_bound:
        if signal_1d[end] < signal_1d[end + 1]:
            break
        end += 1
    if end == right_bound:
        end = peak_idx
        while end < right_bound and signal_1d[end] > baseline:
            end += 1

    return start, end


def detect_ripples(zsignal, fs=2500, threshold=3.0, search_window=None):
    if search_window is None:
        search_window = int(0.15 * fs)
    signal_1d = np.squeeze(zsignal)
    peak_inds, _ = find_peaks(signal_1d, height=threshold)
    rows = []
    for p in peak_inds:
        s, e = _local_minima_boundaries(signal_1d, p, search_window)
        rows.append({
            "peak_idx": p, "peak_amp": signal_1d[p],
            "start_idx": s, "end_idx": e,
            "start_time": s / fs, "peak_time": p / fs,
            "end_time": e / fs, "duration_ms": (e - s) / fs * 1000.0,
        })
    return pd.DataFrame(rows)


def detect_spw_events(lfp_1d, fs=2500, lowcut=5, highcut=40,
                      z_thresh=3.0, min_dur_ms=20, max_dur_ms=400):
    filtered = butter_bandpass_filter(lfp_1d, lowcut, highcut, fs)
    zsig = (filtered - filtered.mean()) / filtered.std()
    peak_idxs, _ = find_peaks(zsig, height=z_thresh)
    min_s, max_s = int(min_dur_ms / 1000 * fs), int(max_dur_ms / 1000 * fs)
    rows = []
    for p in peak_idxs:
        s = p
        while s > 0 and zsig[s] > 0:
            s -= 1
        e = p
        while e < len(zsig) - 1 and zsig[e] > 0:
            e += 1
        dur = e - s
        if min_s <= dur <= max_s:
            rows.append({
                "peak_idx": p, "peak_amp": zsig[p],
                "start_idx": s, "end_idx": e,
                "start_time": s / fs, "peak_time": p / fs,
                "end_time": e / fs, "duration_ms": dur / fs * 1000.0,
            })
    return pd.DataFrame(rows)


def find_spw_overlap(ripple_row, df_spw):
    r_start, r_end, r_peak = ripple_row["start_time"], ripple_row["end_time"], ripple_row["peak_time"]
    overlap = df_spw[(df_spw["end_time"] >= r_start) & (df_spw["start_time"] <= r_end)]
    if len(overlap) == 0:
        return pd.Series({"spw_index": None, "spw_peak_offset_ms": np.nan, "spw_overlap_ms": 0.0})
    closest = overlap.iloc[(overlap["peak_time"] - r_peak).abs().argsort().iloc[0]]
    overlap_ms = max(0, (min(r_end, closest["end_time"]) - max(r_start, closest["start_time"])) * 1000)
    return pd.Series({
        "spw_index": closest.name,
        "spw_peak_offset_ms": (closest["peak_time"] - r_peak) * 1000,
        "spw_overlap_ms": overlap_ms,
    })


def detect_events(lfp, lfp_z, fs=2500, spw_channel_idx=2):
    ripples = detect_ripples(lfp_z, fs=fs)
    spws = detect_spw_events(-lfp[spw_channel_idx], fs=fs)
    ripples[["spw_index", "spw_peak_offset_ms", "spw_overlap_ms"]] = ripples.apply(
        find_spw_overlap, axis=1, df_spw=spws
    )
    ripples["has_spw"] = ripples["spw_index"].notna()
    return ripples


# ---------------------------------------------------------------------------
# Plots

def plot_ripple_triggered_lfp_for_example(
    decision_trace, raw_lfp, ripple_lfp, sw_lfp, ripple_df, fs=2500, xlim=None
):
    n = decision_trace.shape[-1]
    i0 = int(xlim[0] * fs) if xlim else 0
    i1 = int(xlim[1] * fs) if xlim else n
    t_w = np.arange(i0, i1) / fs
    peaks_in = ripple_df.query(f"{xlim[0]} <= peak_time <= {xlim[1]}") if xlim else ripple_df

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t_w, decision_trace[i0:i1], color="black")
    ax1.scatter(
        peaks_in["peak_time"],
        [decision_trace[int(pt * fs)] for pt in peaks_in["peak_time"]],
        color="red", s=20,
    )
    ax1.set_ylabel("z-score")
    ax1.set_title("SWR example")

    ax2.plot(t_w, raw_lfp[i0:i1], color="purple")
    ax2.set_ylabel("Raw LFP (µV)")

    ax3.plot(t_w, ripple_lfp[0, i0:i1], color="blue")
    ax3.set_ylabel("Ripple LFP (µV)")

    ax4.plot(t_w, sw_lfp[0, i0:i1], color="gray")
    ax4.set_ylabel("SW LFP (µV)")
    ax4.set_xlabel("Time (s)")

    for ax in (ax2, ax3, ax4):
        for pt in peaks_in["peak_time"]:
            ax.axvline(pt, color="red", lw=0.5, alpha=0.5)
    for pt in peaks_in["start_time"]:
        ax4.axvline(pt, color="blue", lw=0.5, alpha=0.4)
    for pt in peaks_in["end_time"]:
        ax4.axvline(pt, color="blue", lw=0.5, alpha=0.4)

    plt.tight_layout()
    return fig, axes
