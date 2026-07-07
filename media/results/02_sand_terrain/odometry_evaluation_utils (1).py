from fileinput import filename

import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import warnings
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import os
import pandas as pd
import msgpack
from bisect import bisect_left


mpl.rcParams['text.usetex'] = False
mpl.rcParams['pdf.fonttype'] = 42

# -----------------------------------------------------------------------------
# Plot layout configuration for thesis/report figures
# -----------------------------------------------------------------------------
# The sizes below are in inches. They are chosen to work well when the generated
# PNG/PDF files are included in LaTeX without heavy rescaling.
#
# Change these values here only if you want another report layout. The plotting
# functions below keep using figure_size(...) and TRAJ_FIGSIZE.

PLOT_DPI = 300
# Normal report plots should be tightly cropped so labels are not clipped and
# large blank canvas margins are removed.
SAVE_BBOX_INCHES = 'tight'
# Trajectory plots keep a fixed output shape because their axes use equal scaling.
TRAJECTORY_SAVE_BBOX_INCHES = None
# Applies Matplotlib tight_layout before saving while keeping the fixed figsize.
# This removes unnecessary canvas whitespace and prevents axis labels from being clipped.
REPORT_TIGHT_LAYOUT_PAD = 0.20
SAVE_PAD_INCHES = 0.02
# Some figures contain colorbars/inset axes/equal-aspect axes. Those can trigger
# Matplotlib tight_layout warnings. In that case, use this manual margin fallback.
REPORT_SUBPLOTS_ADJUST = {
    'left': 0.12,
    'right': 0.98,
    'bottom': 0.16,
    'top': 0.97,
}
SUPPRESS_TIGHT_LAYOUT_WARNINGS = True
SHOW_FIGURE_TITLES = False

# Approximate usable text width in an A4 thesis page.
# If your LaTeX document has a different text width, change only this value.
LATEX_TEXT_WIDTH_IN = 6.3
A4_TEXT_HEIGHT_IN = 9.2

# Horizontal report plots: full text width, different height fractions.
FIGSIZE_ONE_FOURTH = (LATEX_TEXT_WIDTH_IN, A4_TEXT_HEIGHT_IN * 0.25)
FIGSIZE_ONE_THIRD  = (LATEX_TEXT_WIDTH_IN, A4_TEXT_HEIGHT_IN * 0.33)
FIGSIZE_HALF       = (LATEX_TEXT_WIDTH_IN, A4_TEXT_HEIGHT_IN * 0.50)
FIGSIZE_FULL       = (LATEX_TEXT_WIDTH_IN, A4_TEXT_HEIGHT_IN * 0.75)

# Trajectory plots are square and intended to fit about half page width.
# Use TRAJECTORY_SIZE_PRESET = 'trajectory_full' if you want full-width trajectory plots.
FIGSIZE_TRAJECTORY_HALF = (LATEX_TEXT_WIDTH_IN * 0.70, LATEX_TEXT_WIDTH_IN * 0.70)
FIGSIZE_TRAJECTORY_FULL = (LATEX_TEXT_WIDTH_IN, LATEX_TEXT_WIDTH_IN)
FIGSIZE_TRAJECTORY_WIDE = (LATEX_TEXT_WIDTH_IN, LATEX_TEXT_WIDTH_IN * 0.70)
TRAJECTORY_SIZE_PRESET = 'trajectory_half'

# Multi-axis/debug plots need extra vertical space.
FIGSIZE_STACKED = FIGSIZE_FULL
FIGSIZE_STACKED_TALL = (LATEX_TEXT_WIDTH_IN, A4_TEXT_HEIGHT_IN * 0.90)
TRAJ_COMPARISON_FIGSIZE = FIGSIZE_HALF

# Keep trajectory tick spacing readable in meters.
# Setting this to None restores Matplotlib default tick spacing.
TRAJECTORY_TICK_SPACING_M = 1.0

# Legacy slide sizes kept for backwards compatibility if you still call them manually.
SLIDE_FIGSIZE_WIDE = (16, 7)
SLIDE_FIGSIZE_SQUARE = (10, 10)
SLIDE_FIGSIZE_LARGE_WIDE = (25, 10)

FONT_SIZE_LEGEND = 7
FONT_SIZE_LABELS = 8
FONT_SIZE_TICKS = 7

# Line thickness for non-trajectory plots.
PLOT_LINE_WIDTH = 0.8
PLOT_MARKER_SIZE = 1.5

# Trajectory legends can easily cover the path in half-page figures.
# Keep this False for report figures; set True only for debug/slide plots.
TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND = False
TRAJECTORY_LEGEND_LOCATION = 'inside_upper_left'  # options: 'outside_bottom', 'inside_upper_left', 'inside_upper_right'
TRAJECTORY_LEGEND_COLUMNS = 1
TRAJECTORY_LEGEND_FONT_SIZE = 6
TRAJECTORY_LEGEND_FRAME_ALPHA = 0.90
TRAJECTORY_SUBPLOT_BOTTOM = 0.28

# Scatter marker area for trajectory start/end points, in points^2.
# Matplotlib default scatter size is usually 36; this is smaller for report figures.
TRAJECTORY_ENDPOINT_MARKER_SIZE = 12

# Line thickness for trajectory plots.
TRAJECTORY_LINE_WIDTH = 0.9
TRAJECTORY_HEADING_VECTOR_WIDTH = 0.0050
TRAJECTORY_COVARIANCE_LINE_WIDTH = 0.9

# Alpha values for trajectory visibility.
# Keep the reference / Vicon trajectory slightly lighter so the compared estimate
# stays easier to see in overlaps. Heading vectors follow the same idea.
TRAJECTORY_REFERENCE_ALPHA = 0.60
TRAJECTORY_COMPARISON_ALPHA = 0.85
TRAJECTORY_BASELINE_ALPHA = 0.85
TRAJECTORY_REFERENCE_HEADING_ALPHA = 0.30
TRAJECTORY_COMPARISON_HEADING_ALPHA = 0.45
TRAJECTORY_BASELINE_HEADING_ALPHA = 0.45

# Keep trajectory plot output shape stable across datasets.
# This prevents long, nearly straight trajectories from becoming a thin strip.
TRAJECTORY_FIX_OUTPUT_SHAPE = True
# None means: use the selected trajectory figure height/width ratio.
# Set to 1.0 if you always want a square plotting box.
TRAJECTORY_AXIS_BOX_ASPECT = None

mpl.rcParams['font.size'] = FONT_SIZE_LABELS
mpl.rcParams['axes.titlesize'] = FONT_SIZE_LABELS
mpl.rcParams['axes.labelsize'] = FONT_SIZE_LABELS
mpl.rcParams['figure.titlesize'] = FONT_SIZE_LABELS
mpl.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
mpl.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
mpl.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
mpl.rcParams['lines.linewidth'] = PLOT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = PLOT_MARKER_SIZE
mpl.rcParams['savefig.pad_inches'] = SAVE_PAD_INCHES

TRAJ_FIGSIZE = {
    'trajectory_half': FIGSIZE_TRAJECTORY_HALF,
    'trajectory_full': FIGSIZE_TRAJECTORY_FULL,
    'trajectory_wide': FIGSIZE_TRAJECTORY_WIDE,
}.get(TRAJECTORY_SIZE_PRESET, FIGSIZE_TRAJECTORY_HALF)

def make_plot_name(plot_kind, reference_label, comparison_label, code, ext="png"):

    if reference_label is None:
        return f"{plot_kind}_{comparison_label}_{code}.{ext}"
    if comparison_label is None:
        return f"{plot_kind}_{reference_label}_{code}.{ext}"
    if reference_label is not None and comparison_label is not None:
        return f"{plot_kind}_{reference_label}_vs_{comparison_label}_{code}.{ext}"
    else:
        raise ValueError("Invalid input for plot name generation, please check the inputs")


def figure_size(scale='one_third'):
    sizes = {
        'one_fourth': FIGSIZE_ONE_FOURTH,
        'one_third': FIGSIZE_ONE_THIRD,
        'half': FIGSIZE_HALF,
        'full': FIGSIZE_FULL,
        'trajectory_half': FIGSIZE_TRAJECTORY_HALF,
        'trajectory_full': FIGSIZE_TRAJECTORY_FULL,
        'trajectory_wide': FIGSIZE_TRAJECTORY_WIDE,
        'stacked': FIGSIZE_STACKED,
        'stacked_tall': FIGSIZE_STACKED_TALL,
    }
    return sizes.get(scale, FIGSIZE_ONE_THIRD)


def _trajectory_tick_spacing_from_span(span):
    if TRAJECTORY_TICK_SPACING_M is None:
        return None
    if span <= 0.5:
        return min(0.10, TRAJECTORY_TICK_SPACING_M)
    if span <= 1.5:
        return min(0.25, TRAJECTORY_TICK_SPACING_M)
    if span <= 3.0:
        return min(0.50, TRAJECTORY_TICK_SPACING_M)
    return TRAJECTORY_TICK_SPACING_M


def _apply_trajectory_tick_spacing(ax):
    x_span = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_span = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    x_spacing = _trajectory_tick_spacing_from_span(x_span)
    y_spacing = _trajectory_tick_spacing_from_span(y_span)
    if x_spacing is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_spacing))
    if y_spacing is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_spacing))



def _set_trajectory_legend(ax, fig=None, include_heading_vectors=None):
    """Apply a compact report-friendly legend for trajectory plots."""
    if include_heading_vectors is None:
        include_heading_vectors = TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND

    handles, labels = ax.get_legend_handles_labels()
    compact_handles = []
    compact_labels = []

    for handle, label in zip(handles, labels):
        if not label or label.startswith('_'):
            continue
        if (not include_heading_vectors) and ('heading vector' in label.lower()):
            continue
        if label in compact_labels:
            continue
        compact_handles.append(handle)
        compact_labels.append(label)

    if not compact_handles:
        return None

    if TRAJECTORY_LEGEND_LOCATION == 'outside_bottom':
        legend = ax.legend(
            compact_handles,
            compact_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=TRAJECTORY_LEGEND_COLUMNS,
            fontsize=TRAJECTORY_LEGEND_FONT_SIZE,
            framealpha=TRAJECTORY_LEGEND_FRAME_ALPHA,
        )
        if fig is not None:
            fig.subplots_adjust(bottom=TRAJECTORY_SUBPLOT_BOTTOM)
        return legend

    legend_loc = {
        'inside_upper_left': 'upper left',
        'inside_upper_right': 'upper right',
        'inside_lower_left': 'lower left',
        'inside_lower_right': 'lower right',
    }.get(TRAJECTORY_LEGEND_LOCATION, 'upper left')

    return ax.legend(
        compact_handles,
        compact_labels,
        loc=legend_loc,
        ncol=TRAJECTORY_LEGEND_COLUMNS,
        fontsize=TRAJECTORY_LEGEND_FONT_SIZE,
        framealpha=TRAJECTORY_LEGEND_FRAME_ALPHA,
    )


def _set_axis_title(ax, title, **kwargs):
    if SHOW_FIGURE_TITLES:
        ax.set_title(title, **kwargs)


def _set_figure_title(fig, title, **kwargs):
    if SHOW_FIGURE_TITLES:
        fig.suptitle(title, **kwargs)


def _apply_report_layout(fig=None, pad=None):
    """Apply compact report layout without changing plot data or axes logic.

    This is intentionally called before saving. Without it, fixed-size report
    figures can keep large default margins and can clip the x-axis label.

    Some figures, for example figures with colorbars, inset axes, or special
    axes, are not fully compatible with tight_layout. Matplotlib reports this
    as a UserWarning, not an exception. Catch that warning and fall back to a
    simple manual subplot adjustment so the warning does not pollute batch logs.
    """
    if fig is None:
        fig = plt.gcf()
    if pad is None:
        pad = REPORT_TIGHT_LAYOUT_PAD

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            if SUPPRESS_TIGHT_LAYOUT_WARNINGS:
                warnings.simplefilter("always", UserWarning)
            fig.tight_layout(pad=pad)

        incompatible_tight_layout = any(
            "not compatible with tight_layout" in str(warning.message)
            for warning in caught_warnings
        )

        if incompatible_tight_layout:
            fig.subplots_adjust(**REPORT_SUBPLOTS_ADJUST)

    except Exception:
        fig.subplots_adjust(**REPORT_SUBPLOTS_ADJUST)


def _save_report_figure(path, fig=None, bbox_inches=SAVE_BBOX_INCHES):
    """Save a report figure after applying layout.

    Keeping bbox_inches=None preserves the configured figsize/output shape,
    while tight_layout prevents clipped labels.
    """
    if fig is None:
        fig = plt.gcf()
    _apply_report_layout(fig)
    fig.savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches=bbox_inches,
        pad_inches=SAVE_PAD_INCHES,
    )


def _trajectory_target_box_aspect():
    """
    Return target axes height/width ratio for trajectory plots.
    This is layout-only and is used to keep the saved output shape stable.
    """
    if TRAJECTORY_AXIS_BOX_ASPECT is not None:
        return float(TRAJECTORY_AXIS_BOX_ASPECT)

    width, height = TRAJ_FIGSIZE
    if width <= 0:
        return 1.0
    return float(height) / float(width)


def _set_trajectory_view(ax, positions, left_pad=0.06, right_pad=0.06, bottom_pad=0.06, top_pad=0.06):
    """
    Set trajectory limits without changing trajectory data.

    When TRAJECTORY_FIX_OUTPUT_SHAPE=True, the shorter axis limit is expanded
    so that equal data aspect does not make the plot box collapse into a thin
    strip for long, almost-straight trajectories.
    """

    valid_positions = []
    for position in positions:
        if position is None:
            continue

        position_array = np.asarray(position)
        if position_array.ndim != 2 or position_array.shape[0] == 0 or position_array.shape[1] < 2:
            continue

        valid_positions.append(position_array[:, :2])

    if not valid_positions:
        return

    stacked_positions = np.vstack(valid_positions)
    x_min = stacked_positions[:, 0].min()
    x_max = stacked_positions[:, 0].max()
    y_min = stacked_positions[:, 1].min()
    y_max = stacked_positions[:, 1].max()

    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    x_lim_min = x_min - left_pad * x_span
    x_lim_max = x_max + right_pad * x_span
    y_lim_min = y_min - bottom_pad * y_span
    y_lim_max = y_max + top_pad * y_span

    if TRAJECTORY_FIX_OUTPUT_SHAPE:
        box_aspect = _trajectory_target_box_aspect()  # axes height / axes width

        x_center = 0.5 * (x_lim_min + x_lim_max)
        y_center = 0.5 * (y_lim_min + y_lim_max)
        x_range = max(x_lim_max - x_lim_min, 1e-9)
        y_range = max(y_lim_max - y_lim_min, 1e-9)

        # For equal metric scaling, y_range / x_range should match box_aspect.
        target_y_range = x_range * box_aspect

        if y_range < target_y_range:
            y_range = target_y_range
        else:
            x_range = y_range / box_aspect

        x_lim_min = x_center - 0.5 * x_range
        x_lim_max = x_center + 0.5 * x_range
        y_lim_min = y_center - 0.5 * y_range
        y_lim_max = y_center + 0.5 * y_range

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(box_aspect)

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    _apply_trajectory_tick_spacing(ax)

def _trajectory_role_color(label, role, fallback='red'):
    normalized_label = (label or '').lower()

    if role == 'reference':
        return 'blue'
    if role == 'baseline':
        return 'green'

    if any(keyword in normalized_label for keyword in ('baseline', 'skid')):
        return 'green'
    if any(keyword in normalized_label for keyword in ('vicon', 'ground truth', 'ground_truth', 'reference')):
        return 'blue'
    if any(keyword in normalized_label for keyword in ('contact_velocity', 'cvo')):
        return 'red'

    return fallback

def dict_generation(data, data_name: str = ''):

    data_dict = {}

    data_dict['timestamp'] = np.array(data[data_name]['time.microseconds'])
    data_dict['position'] = np.array(data[data_name]['position.data'])

    im= np.array(data[data_name]['orientation.im'])
    re = np.array(data[data_name]['orientation.re'])

    # merge the two arrays
    data_dict['orientation_quat'] = np.column_stack((re, im)) # (w, x, y, z)

    return data_dict

def nearest_match(sorted_times, target):
    i = bisect_left(sorted_times, target)
    candidates = []
    if i > 0:
        candidates.append(sorted_times[i - 1])
    if i < len(sorted_times):
        candidates.append(sorted_times[i])
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - target))

def sync_data_streams_timestamps(source_timestamps, target_timestamps, max_time_diff=10000):
    """
    Sync source timestamps to target timestamps using nearest-neighbor matching.

    Parameters
    ----------
    source_timestamps : list or np.ndarray
        Usually the higher-rate stream.

    target_timestamps : list or np.ndarray
        Usually the lower-rate stream.

    max_time_diff : float
        Maximum allowed timestamp difference in microseconds.

    Returns
    -------
    synced_source : list
        Source timestamps matched to target timestamps.

    synced_target : list
        Target timestamps that found a valid source match.
    """

    source_timestamps = sorted([int(t) for t in source_timestamps])
    target_timestamps = sorted([int(t) for t in target_timestamps])

    synced_source = []
    synced_target = []

    source_idx = 0
    num_source = len(source_timestamps)

    for target_time in target_timestamps:

        # Move source index forward until source is inside possible window
        while (
            source_idx < num_source
            and source_timestamps[source_idx] < target_time - max_time_diff
        ):
            source_idx += 1

        candidates = []

        if source_idx < num_source:
            candidates.append(source_idx)

        if source_idx > 0:
            candidates.append(source_idx - 1)

        if not candidates:
            continue

        best_idx = min(
            candidates,
            key=lambda idx: abs(source_timestamps[idx] - target_time)
        )

        best_time = source_timestamps[best_idx]

        if abs(best_time - target_time) <= max_time_diff:
            synced_source.append(best_time)
            synced_target.append(target_time)

            # Prevent reusing the same source sample
            source_idx = best_idx + 1

    return synced_source, synced_target


def sync_data_streams(odom_data_sync, gt_data_sync, max_time_diff=1e5):
    odom_time = sorted(odom_data_sync.keys())
    gt_time = sorted(gt_data_sync.keys())
    
    print("shape of odom_time: ", len(odom_time))
    print("shape of gt_time: ", len(gt_time))

    synced_odom = []
    synced_gt = []

    odom_idx = 0
    for t in gt_time:
        while odom_idx < len(odom_time) and odom_time[odom_idx] < t - max_time_diff:
            odom_idx += 1

        best = nearest_match(odom_time[odom_idx:], t)
        if best is not None and abs(best - t) <= max_time_diff:
            synced_gt.append(t)
            synced_odom.append(best)
            odom_idx = odom_time.index(best) + 1
            
    if len(synced_odom) == 0 or len(synced_gt) == 0:
        print("No synced data points found, please check the data and the max_time_diff parameter")
        raise ValueError("No synced data points found")

    return synced_odom, synced_gt

# def sync_data_streams(odom_data_sync, gt_data_sync, max_time_diff=1e5):
#     '''
#     return the odometry and ground truth timestamps after syncing
#     :param odom_data_sync: odometry data with timestamps as keys
#     :param gt_data_sync: ground truth data with timestamps as keys
#     :return: odometry and ground truth data timestamps with the closest starting timestamp as the starting point

#     Assumption: the odometry data stream is faster than the ground truth data stream
#     '''
#     odom_time = list(odom_data_sync.keys())
#     gt_time = list(gt_data_sync.keys())

#     odom_start_idx, gt_start_idx = None, None
#     # find the starting point of the odometry data stream

#     stop = False
#     for i in range(len(odom_time)):
#         if stop:
#             break
#         for j in range(len(gt_time)):
#             if abs(odom_time[i] - gt_time[j]) > 2e6: # if gap is more than 2 seconds move the starting odom time to the next
#                 break
#             elif abs(odom_time[i] - gt_time[j]) < 2e6: # if the time difference is less than milliseconds then we have found the close enough starting point
#                 print("starting_point_found")
#                 odom_start_idx = i
#                 gt_start_idx = j
#                 stop = True # stop the outer loop when the starting point is found
#                 break
#             else:
#                 continue
    
#     if odom_start_idx is None or gt_start_idx is None:
#         print("comparision timestamp range: ", odom_time[0], " to ", odom_time[-1])
#         print("Reference timestamp range: ", gt_time[0], " to ", gt_time[-1])
#         print("start_diff: ", abs(odom_time[0] - gt_time[0]) / 1e6, "s", " end_diff: ", abs(odom_time[-1] - gt_time[-1]) / 1e6, "s")

#         print("[DEBUG] reference starting timestamp : ", gt_time[0] )
#         print("[DEBUG] comparison starting timestamp: ", odom_time[0])
#         print("No valid starting point found")
#         raise ValueError("No starting point found")
#     else:
#         print("Begin syncing data streams")
#         s,a = odom_time[odom_start_idx:], gt_time[gt_start_idx:] # s is sensor data and a is actual or true data

#         sync_s = []
#         sync_a = []

#         i, j = 0, 0
#         while i < len(s) and j < len(a):
#             diff = abs(a[j] - s[i])
            
#             if diff < max_time_diff: # less than a 0.1 seconds
#                 sync_a.append(a[j])
#                 sync_s.append(s[i])
#                 i += 1
#                 j += 1
#             elif a[j] < s[i]:
#                 j += 1
#             else:
#                 i += 1
                
#         print("sync progress complete")
#         if len(sync_a) == 0 or len(sync_s) == 0:
#             print("No synced data points found, please check the data and the max_time_diff parameter")
#             raise ValueError("No synced data points found")
        
#     return sync_s, sync_a


def relational_msg_to_dict(evaluation_dir_path, log_file_name, sample_name):

    """
    Convert the msgpack file to a dictionary with timestamps as keys and position and orientation as values.

    Args:
        evaluation_dir_path (str): Path to the eval uation directory.
        log_file_name (str): Name of the log file.
        sample_name (str): Name of the sample.
    Returns:
        dict: Dictionary with timestamps as keys and position and orientation as values.
    """

    log_file_path = os.path.join(evaluation_dir_path, log_file_name)

    data= dict_generation(pd.DataFrame(msgpack.load(open(log_file_path, "rb"))), sample_name)

    # we use the timestamps as keys and the position and orientation as values
    data_dict = dict()

    for i in range(len(data['timestamp'])):
        data_dict[data['timestamp'][i]] = {'position': data['position'][i], 'orientation': data['orientation_quat'][i]}

    return data_dict

def fd(p1: np.ndarray, p2: np.ndarray, dim=2):
    '''
    The FindDistance calculate the euclidean distance between two 3D points
    can only using the x and y coordinates of the points or x, y and z coordinates
    :param p1: point 1
    :param p2: point 2
    :param dim: dimension of the points
    :return: euclidean distance between the two points
    '''

    if dim == 2:
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    elif dim == 3:
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    else:
        print("Enter valid dimension, either 2 or 3")
        raise ValueError("Invalid dimension")
    return distance

def fa(a1, a2):
    '''
    The FindAngle function calculates the euler angles difference for two set of euler angles
    :param a1: euler angle array or 1D data
    :param a2: euler angle array or 1D data
    :return: array of euler angle in radians
    '''

    a1 = np.array(a1)
    a2 = np.array(a2)

    if a1.shape != a2.shape:
        print("The inputs must have the same shape")
        raise ValueError("The inputs must have the same shape")
    elif a1.ndim == 1:
        angle_diff = a1 - a2
        normalized_angle_diff = np.array([normalize_angle(diff) for diff in angle_diff])
    else:
        angle_diff = a1 - a2
        normalized_angle_diff = np.apply_along_axis(normalize_angle, axis=1, arr=angle_diff)

    return normalized_angle_diff

def fga(q1, q2=None):
    """
    Find geodesic rotation angle.

    If q2 is provided:
        returns the geodesic angle between q1 and q2

    If q2 is None:
        returns the rotation angle represented by q1

    The angular_distance_geodesic function simply called as fga() find_geodesic_angle between two quaternions
    q1, q2: quaternions in the form of [qw, qx, qy, qz]
    : param q1: first quaternion
    : param q2: second quaternion
    Returns the angular distance in radians between two quaternions using the geodesic method.

    Refer: R. Hartley & C. Trumpf, “Rotation Averaging”, International Journal of Computer Vision, vol. 103, no. 3, 2013, p. 267-305 under Relation to the angle-axis formulation subsection.
    """

    q1 = np.asarray(q1, dtype=float)
    q1 = q1 / np.linalg.norm(q1)

    if q2 is None:
        # angle of a single rotation
        w = np.clip(np.abs(q1[0]), -1.0, 1.0)
        return 2.0 * np.arccos(w)

    q2 = np.asarray(q2, dtype=float)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)

    return 2.0 * np.arccos(dot)

def calculate_travel_distance(array_position, dim=3):
    """
    Calculate the travel distance of the robot using the position data

    :param array_position: 2D or 3D array of position data
    :return: travel distance in meters
    """
    array_position = np.asarray(array_position)

    # Only use the first 'dim' coordinates
    positions = array_position[:, :dim]

    # Compute differences between consecutive points
    diffs = np.diff(positions, axis=0)

    # Euclidean distances of each displacement
    distances = np.linalg.norm(diffs, axis=1)

    # Total travel distance
    return distances.sum()

def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    :param angle: Angle in radians.
    :return: Normalized angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_euler(q_w, q_x, q_y, q_z, debug=False):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw) in radians.
    check if the quaternion is normalized, if not normalize it
    :param q_w: real component of the quaternion
    :param q_x: imaginary x component of the quaternion
    :param q_y: imaginary y component of the quaternion
    :param q_z: imaginary z component of the quaternion

    return: [roll, pitch, yaw] normalized to [-pi, pi] in radians
    Source: "https://discuss.luxonis.com/d/5453-how-to-convert-quaternions-to-pitchrollyaw/2"
    """

    #check if the quaternion is normalized
    norm = math.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
    if abs(norm - 1) > 1e-6:
        if debug:
            print("Not a unit quaternion, normalizing...")
            print(f"Norm: {norm}")
        q_w /= norm
        q_x /= norm
        q_y /= norm
        q_z /= norm
    else:
        if debug:
            print("Unit quaternion, no need to normalize.")
        pass

    # Roll (x-axis rotation)
    roll = math.atan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2))
    
    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (q_w * q_y - q_z * q_x))
    
    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))

    roll = normalize_angle(roll)
    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    
    return [roll, pitch, yaw]

def median_filter(data, window_size):
    """
    Apply a median filter to the data with a specified window size.
    Args:
        data: The input array data 1D to be filtered.
        window_size (int): The size of the median filter window.
    Returns:
        filtered : The filtered data array of 1D data.
    """
    padded = np.pad(data, (window_size//2,), mode='edge')
    filtered = [
        np.median(padded[i:i+window_size])
        for i in range(len(data))
    ]
    return np.array(filtered)

def elapsed_time_in_seconds(timestamps):
    """
    Convert timestamps in microseconds to elapsed time in seconds.
    :param timestamps_us: List of timestamps in microseconds.
    :return: List of elapsed time in seconds.
    """
    datetimes = [datetime.fromtimestamp(ts / 1e6) for ts in timestamps]
    base_time = datetimes[0].timestamp()
        
    return [dt.timestamp() - base_time for dt in datetimes]

def remove_outliers_from_yaw_data(vicon_yaw_, odom_yaw_, vicon_timestamp_, filter_threshold=100, debug=False, show=False):
    """
    Function to remove outliers from the yaw data
    :param vicon_yaw_: The VICON yaw data (in radians)
    :param odom_yaw_: The odometry yaw data (in radians)
    :param vicon_timestamp_: The timestamps of the VICON data (in microseconds)
    :param filter_threshold (in deg): The change of angle above which the value change is considered as outlier
                                      The default value is 100 degrees, since the robot cannot rotate that fast in a short time

    :return: The filtered VICON and odometry data (in radians)
    :return: The filtered vicon timestamps
    :return: The indices where the yaw changes significantly
    
    """

    filtered_vicon_data ,filtered_odom_data, filtered_vicon_timestamp = vicon_yaw_, odom_yaw_, vicon_timestamp_

    if len(vicon_yaw_) != len(odom_yaw_) :
        raise ValueError("The length of the VICON and odometry data do not match")
    if len(vicon_yaw_) != len(vicon_timestamp_) :
        raise ValueError("The length of the VICON and timestamps do not match")
    
    #unwrap the yaw angles to remove discontinuities
    vicon_yaw = np.unwrap(np.array(vicon_yaw_))
    odom_yaw = np.unwrap(np.array(odom_yaw_))

    # Align the yaw angles if there is a shift of more than pi
    #TODO: After unwrapping, sometimes the yaw angles are shifted by 2*pi, need to check why and decide if this step is needed
    if abs(vicon_yaw[0] - odom_yaw[0]) > np.pi and abs(vicon_yaw[0] - odom_yaw[0]) < 2 * np.pi:
        if vicon_yaw[0] < odom_yaw[0]:
            vicon_yaw += 2 * np.pi
        else:
            vicon_yaw -= 2 * np.pi
    else:
        if debug: 
            print("No yaw shift detected")

    # find the indices where the yaw changes significantly to remove the outliers
    v =np.degrees(vicon_yaw)
    idx_change = np.where(np.abs(np.diff(v)) > filter_threshold)[0] # returns the index before the jump if the jump is more than 100

    if debug:
        print("The indices where the vicon yaw changes significantly are: ", idx_change)
    # remove the outliers
    index_to_remove = []
    if len(idx_change) >= 2:
        for start, end in zip(idx_change[:-1], idx_change[1:]):
            if abs(vicon_yaw[start] - vicon_yaw[end]) > np.deg2rad(filter_threshold):# Check if the consecutive indices has a jump TODO: Need further testing, This might break with rate of rotation is high 
                index_to_remove.append(np.arange(start, end+1))
            else:
                pass
    elif len(idx_change) == 0:
        print("No significant change in yaw detected, no outliers to remove")
    else:
        raise ValueError("The number of indices where the yaw changes significantly is only 1, please check the data to see if there are any actual outliers")

    if debug:
        print("The indices to be removed are: ", index_to_remove)
        print("The number of indices to be removed are: ", len(index_to_remove))
    # merge the multiple nested arrays into a single array
    if index_to_remove:
        index_to_remove = np.concatenate(index_to_remove)
    else:
        index_to_remove = np.array([])

    # remove the outliers from the data
    if np.array(index_to_remove).size > 0:
        filtered_vicon_data = np.delete(filtered_vicon_data, index_to_remove)
        filtered_odom_data = np.delete(filtered_odom_data, index_to_remove)
        filtered_vicon_timestamp = np.delete(vicon_timestamp_, index_to_remove)
    else:
        print("No outlier need to be removed")

    if debug:
        plt.plot(vicon_timestamp_, np.degrees(vicon_yaw_), label="VICON Yaw")
        plt.plot(filtered_vicon_timestamp, np.degrees(filtered_vicon_data), label="Filtered VICON Yaw")
        # plt.title("[DEBUG] VICON Yaw data before and after filtering")
        plt.xlabel("Timestamp")
        plt.ylabel("Yaw angle in degrees")
        plt.legend()
        if show:
            plt.show()
        else:
            print("No plot shown, set show=True to show the plot")

    return filtered_vicon_data, filtered_odom_data, filtered_vicon_timestamp, idx_change, index_to_remove

def calculate_absolute_pose_error(ground_truth, odometry, code='unknown',
                                  save=False,
                                  show=False,
                                  plots_path='/opt/workspace/datasets/',
                                  odom_type='unknown',
                                  comparison_label=None,
                                  reference_label=None):
    """
    Calculate the Absolute Pose Error (APE) between the ground truth and odometry data.
    Compares every pose.
    :param ground_truth: np.ndarray -> [position(3D), orientation(quaternions)]  
    :param odometry:     np.ndarray -> [position(3D), orientation(quaternions)] 
    :return error: dict  -> {'position_errors': list, 'geodesic_angle_errors': list(degrees),
                              'position_errors_rmse': float, 'geodesic_angle_errors_rmse': list(degrees)}
    """

    trans_error_rmse, rot_error_rmse = None, None

    #check if the ground truth and odometry is same shape
    if not ((ground_truth.shape) == (odometry.shape) and ground_truth.shape[1] == 7): # 3 for position and 4 for quaternion
        raise ValueError("Check the shapes for the input arrays, the input array shapes should be same and should have seven column. \n"  
                         f"Got ground_truth: {ground_truth.shape}, odometry: {odometry.shape}")
    
    error_trans, error_rot = [], []
    distances = [0.0]
    for idx in range(len(ground_truth)):
        # translation error
        error_trans.append(fd(ground_truth[idx][:3] , odometry[idx][:3]))
        # rotation error
        error_rot.append(fga(ground_truth[idx, 3:7], odometry[idx, 3:7]))

        # accumulate distance along ground truth path
        if idx < len(ground_truth) - 1:
            distances.append(distances[-1] + fd(ground_truth[idx, :3], ground_truth[idx+1, :3]))

    error_rot = np.degrees(error_rot)  # convert to degrees for better interpretability
    trans_error_rmse = float(np.sqrt(np.mean(np.square(error_trans))))
    rot_error_rmse = float(np.sqrt(np.mean(np.square(error_rot))))

    # labels (backwards compatible)
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'

    # plot the APE results
    plt.figure(figsize=FIGSIZE_ONE_THIRD)
    plt.plot(distances, error_trans, label=f"{comp_label} translational Error", marker='.', markersize=2)
    # plt.title(f"{comp_label} Absolute Translational Error : {code}")
    plt.xlabel("Travel Distance [m]")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.grid()
    _apply_report_layout()
    if save:
        _save_report_figure(f'{plots_path}/APE_translational_{comp_label}_{code}.png')
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=FIGSIZE_ONE_THIRD)
    plt.plot(distances, error_rot, label=f"{comp_label} rotational Error", marker='.', markersize=2)
    # plt.title(f"{comp_label} Absolute Rotational Error : {code}")
    plt.xlabel("Travel Distance [m]")
    plt.ylabel("Error [deg]")
    plt.legend()
    plt.grid()
    _apply_report_layout()
    if save:
        _save_report_figure(f'{plots_path}/APE_rotational_{comp_label}_{code}.png')
    if show:
        plt.show()
    plt.close()  # Close the plot to free up memory

    return {
        'position_errors': error_trans,
        'geodesic_angle_errors': error_rot,
        'position_errors_rmse': trans_error_rmse,
        'geodesic_angle_errors_rmse': rot_error_rmse,
    }

def calculate_relative_pose_error(ground_truth, odometry, distance_threshold=0.5):
    """
    Calculate the Relative Pose Error (RPE) between the ground truth and odometry data by sliding window method based on distance.
    :param ground_truth: np.ndarray -> [position(3D), orientation(quaternion)]
    :param odometry: np.ndarray -> [position(3D), orientation(quaternion)]
    :param distance_threshold: float -> window length in meters (default 0.5)
    :return : trans_error_per_distance: list of translation error in meters
              rot_error_per_distance: list of rotation error in radians
              cumsum_dist: cumulative distance array for the trajectory
    """
    if not ((ground_truth.shape) == (odometry.shape) and ground_truth.shape[1] == 7):
        raise ValueError("Check the shapes for the input arrays; expected Nx7 arrays")

    # compute distances between consecutive ground-truth poses
    diffs = np.diff(ground_truth[:, :3], axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    # cumulative distance including initial zero
    cumsum_dist = np.concatenate(([0.0], np.cumsum(distances)))

    start_idx = 0
    trans_error_per_distance = []
    rot_error_per_distance = []

    while start_idx < len(ground_truth) - 1:
        target = cumsum_dist[start_idx] + distance_threshold
        end_idx = np.searchsorted(cumsum_dist, target)

        if end_idx >= len(ground_truth) - 1:
            break

        # compute relative poses
        dp_gt, dq_gt = calculate_relative_pose(
            ground_truth[start_idx][:3], ground_truth[start_idx][3:7],
            ground_truth[end_idx][:3],   ground_truth[end_idx][3:7]
        )

        dp_odom, dq_odom = calculate_relative_pose(
            odometry[start_idx][:3], odometry[start_idx][3:7],
            odometry[end_idx][:3],   odometry[end_idx][3:7]
        )

        # RPE pose error
        dp_err = dp_odom - dp_gt
        dq_err = multiply_quat(conjugate_quat(dq_gt), dq_odom)

        # Split errors
        trans_error_per_distance.append(np.linalg.norm(dp_err))
        rot_error_per_distance.append(np.linalg.norm(fga(dq_err)))

        start_idx += 1

    return trans_error_per_distance, rot_error_per_distance, cumsum_dist

def conjugate_quat(q):
    """
    Quaternion conjugate
    q: np.array [w, x, y, z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def multiply_quat(q1, q2):
    """
    Quaternion multiplication q = q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def calculate_relative_quaternion(q_start, q_end):
    """
    Calculate the relative rotation quaternion from q_start to q_end
    q_relative = q_end ⊗ q_start_conjugate
    """
    q_start_conj = conjugate_quat(q_start)
    return multiply_quat(q_end, q_start_conj)

def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector v by a quaternion q
    v: np.array [x, y, z]
    q: np.array [w, x, y, z]
    """
    # v' = q ⊗ v_quat ⊗ q_conjugate
    q_conj = conjugate_quat(q)
    v_quat = np.array([0, v[0], v[1], v[2]])
    rotated_v_quat = multiply_quat(multiply_quat(q, v_quat), q_conj)
    return rotated_v_quat[1:]

def calculate_relative_pose(p1, q1, p2, q2):
    # relative translation
    dp_world = p2 - p1

    dp_local = rotate_vector_by_quaternion(dp_world, conjugate_quat(q1))

    # relative rotation
    dq_local = multiply_quat(conjugate_quat(q1), q2)
    return dp_local, dq_local

def find_indices_at_distance_intervals(positions, distance_interval=1.0):

    """
    Get the index in the positions list where the travel distance from the start_index exceeds the distance_threshold
    :param positions: np.ndarray -> [position(3D)]
    :param distance_interval: float -> distance interval in meters
    :return: index: int -> index where the distance exceeds the threshold
    """
    positions = np.array(positions)

    # Euclidean distances between consecutive points
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # Cumulative distance from start_index
    cum_distances = np.cumsum(distances)
    
    total_distance = cum_distances[-1]

    # Target distances: 1m, 2m, 3m, ... up to total distance
    targets = np.arange(distance_interval, total_distance, distance_interval)

    # For each target, find the first index where cumulative distance >= target
    # +1 because cum_distances[i] corresponds to positions[i+1]
    indices = [np.searchsorted(cum_distances, t) + 1 for t in targets]

    # Always include index 0 as the starting marker
    indices = [0] + indices

    return indices

def rotate_trajectory_yaw(positions, orientation_quats, angle, rotate_position=True):
    """
    Rotate the trajectory by 180 degrees around the Z-axis
    :param positions: Nx3 array of positions
    :param orientation_quats: Nx4 array of quaternions (w, x, y, z)
    :return: rotated_positions, rotated_quaternions
    """
    angle = np.deg2rad(angle)  # Convert angle to radians

    rotation_quat_angle = R.from_rotvec(np.array([0, 0, angle]))
    if rotate_position:
        origin = positions[0]  # Anchor point = trajectory start , This is so that a non zero starting point can also be rotated around the Z-axis
        centered = positions - origin          # Translate to origin
        rotated = rotation_quat_angle.apply(centered)  # Rotate around origin
        rotated_positions = rotated + origin   # Translate back
    else:
        rotated_positions = positions

    # reorder the quaternions from (w,x,y,z) to (x,y,z,w)
    orientation_quats = orientation_quats[:, [1, 2, 3, 0]]

    # Create Rotation objects from the (x,y,z,w) quaternions
    original_rotations = R.from_quat(orientation_quats)
    
    # Apply the 180-degree rotation
    rotated_rotations = rotation_quat_angle * original_rotations
    rotated_quaternions_xyzw = rotated_rotations.as_quat()  # Get (x, y, z, w)

    # Reorder the quaternions back to (w, x, y, z)
    rotated_quaternions_wxyz = rotated_quaternions_xyzw[:, [3, 0, 1, 2]]
    return rotated_positions, rotated_quaternions_wxyz

def rotate_trajectory_yaw_euler(positions, orientation_euler, angle):
    """
    Rotate the trajectory by 180 degrees around the Z-axis using Euler angles
    :param positions: Nx3 array of positions
    :param orientation_euler: Nx3 array of Euler angles (roll, pitch, yaw)
    :param angle: angle to rotate the trajectory by in degrees
    :return: rotated_positions, rotated_euler_angles
    """
    angle = np.deg2rad(angle)  # Convert angle to radians

    # Rotate the positions
    rotation_matrix = R.from_euler('z', angle).as_matrix()
    rotated_positions = positions @ rotation_matrix.T

    # Rotate the yaw angle
    rotated_yaw = orientation_euler[:, 2] + angle
    rotated_yaw = np.mod(rotated_yaw + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]

    # Create new Euler angles with unchanged roll and pitch
    rotated_euler_angles = orientation_euler.copy()
    rotated_euler_angles[:, 2] = rotated_yaw

    return rotated_positions, rotated_euler_angles



def align_initial_pose_se3(
    odom_positions,
    odom_orientations,
    gt_positions,
    gt_orientations,
):
    """
    Align an odometry trajectory to the ground truth frame using
    the initial pose.

    Parameters
    ----------
    odom_positions : (N,3)
    odom_orientations : (N,4) quaternions (w,x,y,z)

    gt_positions : (N,3)
    gt_orientations : (N,4) quaternions (w,x,y,z)

    Returns
    -------
    aligned_positions : (N,3)
    aligned_orientations : (N,4) (w,x,y,z)
    """

    # --- initial rotation ---
    q_delta = multiply_quat(
        gt_orientations[0],
        conjugate_quat(odom_orientations[0])
    )

    # scipy wants (x,y,z,w)
    q_delta_xyzw = [
        q_delta[1],
        q_delta[2],
        q_delta[3],
        q_delta[0]
    ]

    R_delta = R.from_quat(q_delta_xyzw)

    # --- initial translation ---
    t_delta = (
        gt_positions[0]
        - R_delta.apply(odom_positions[0])
    )

    # --- transform all positions ---
    aligned_positions = (
        R_delta.apply(odom_positions)
        + t_delta
    )

    # --- transform all orientations ---
    aligned_orientations = np.array([
        multiply_quat(q_delta, q)
        for q in odom_orientations
    ])

    return aligned_positions, aligned_orientations


def align_initial_position(odom_position, vicon_position):
    """
    Align the initial position(x and y) of the odometry data to the ground truth data by translating the odometry trajectory
    :param odom_position: Nx2 or Nx3 array of odometry positions
    :param vicon_position: Nx2 or Nx3 array of ground truth positions
    :return: aligned_odom_position: same shape as odom_position with x,y aligned to vicon start 
    """
    # adjust x,y and z
    translation_xyz = vicon_position[0] - odom_position[0]
    aligned_odom_position = np.array(odom_position, copy=True)
    aligned_odom_position += translation_xyz
    return aligned_odom_position

def align_initial_orientation(odom_orientation, vicon_orientation):
    """
    Align the initial orientation of the odometry data to the ground truth data by applying a fixed quaternion rotation.
    :param odom_orientation: Nx4 array of odometry quaternions in (w, x, y, z)
    :param vicon_orientation: Nx4 array of ground truth quaternions in (w, x, y, z)
    :return: aligned_odom_orientation: same shape as odom_orientation with the first orientation aligned to vicon
    """
    odom_orientation = np.asarray(odom_orientation)
    vicon_orientation = np.asarray(vicon_orientation)

    if odom_orientation.shape != vicon_orientation.shape:
        raise ValueError(
            "The odometry and ground truth orientations must have the same shape"
        )
    if odom_orientation.ndim != 2 or odom_orientation.shape[1] != 4:
        raise ValueError("Orientation inputs must be Nx4 arrays in (w, x, y, z) format")

    initial_delta_quat = multiply_quat(vicon_orientation[0], conjugate_quat(odom_orientation[0]))

    aligned_odom_orientation = np.array(odom_orientation, copy=True)
    aligned_odom_orientation = np.array([
        multiply_quat(initial_delta_quat, quat)
        for quat in aligned_odom_orientation
    ])

    return aligned_odom_orientation

def plot_position_comparison(x_axis, odom_position, vicon_position, code, description='unknown' ,axis='z', save=False, plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
    plt.figure(figsize=figure_size('one_third'))
    x_axis = elapsed_time_in_seconds(x_axis) # convert the timestamps to elapsed time in seconds for better visualization
    # label selection: prefer explicit labels; fall back to 'comparison'/'reference'
    comp_label = comparison_label if comparison_label is not None else 'comparison'
    ref_label = reference_label if reference_label is not None else 'reference'
    if axis == 'x':
        plt.plot(x_axis, odom_position[:, 0], label=f'{comp_label} X', color='red')
        plt.plot(x_axis, vicon_position[:, 0], label=f'{ref_label} X', color='blue')
    elif axis == 'y':
        plt.plot(x_axis, odom_position[:, 1], label=f'{comp_label} Y', color='red')
        plt.plot(x_axis, vicon_position[:, 1], label=f'{ref_label} Y', color='blue')
    elif axis == 'z':
        plt.plot(x_axis, odom_position[:, 2], label=f'{comp_label} Z', color='red')
        plt.plot(x_axis, vicon_position[:, 2], label=f'{ref_label} Z', color='blue')


    # plt.title(f'Position Comparison ({axis.upper()} Axis) for {comp_label} vs {ref_label} - {code}')
    plt.xlabel('Time in seconds')
    plt.ylabel(f'{axis.upper()} Position (m)')
    plt.legend()
    plt.grid()

    if save:
        filename = make_plot_name(
        "position_comparison",
        reference_label,
        comparison_label,
        code)
        _save_report_figure(f'{plots_path}/{f"{axis}_{filename}"}')
    else:
        print(f"Showing position comparison for {axis} axis with description: {description}_{code}")
        _apply_report_layout()
        plt.show()
    plt.close() 

def plot_orientation_comparision(x_axis, odom_orientation, vicon_orientation, code, description='unknown', axis='yaw', save=False,  plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
    plt.figure(figsize=figure_size('one_third'))
    x_axis = elapsed_time_in_seconds(x_axis) # convert the timestamps to elapsed time in seconds for better visualization

    odom_orientation = np.unwrap(np.array(odom_orientation), axis=0)
    vicon_orientation = np.unwrap(np.array(vicon_orientation), axis=0)

    # This is to keep the plot within the range of -180 to 180 degrees, 
    if np.mean(odom_orientation[:, 2]) >  np.pi:
        odom_orientation[:, 2] -= 2 * np.pi
    if np.mean(odom_orientation[:, 2]) < -np.pi:
        odom_orientation[:, 2] += 2 * np.pi
    if np.mean(vicon_orientation[:, 2]) >  np.pi:
        vicon_orientation[:, 2] -= 2 * np.pi
    if np.mean(vicon_orientation[:, 2]) < -np.pi:
        vicon_orientation[:, 2] += 2 * np.pi

    odom_orientation = np.degrees(odom_orientation)
    vicon_orientation = np.degrees(vicon_orientation)

    comp_label = comparison_label if comparison_label is not None else description if description is not None else 'Odometry'
    comp_label = comparison_label if comparison_label is not None else 'comparison'
    ref_label = reference_label if reference_label is not None else 'reference'
    if axis == 'yaw':
        plt.plot(x_axis, odom_orientation[:, 2], label=f'{comp_label} Yaw', color='red')
        plt.plot(x_axis, vicon_orientation[:, 2], label=f'{ref_label} Yaw', color='blue')
    elif axis == 'pitch':
        plt.plot(x_axis, odom_orientation[:, 1], label=f'{comp_label} Pitch', color='red')
        plt.plot(x_axis, vicon_orientation[:, 1], label=f'{ref_label} Pitch', color='blue')
    elif axis == 'roll':
        plt.plot(x_axis, odom_orientation[:, 0], label=f'{comp_label} Roll', color='red')
        plt.plot(x_axis, vicon_orientation[:, 0], label=f'{ref_label} Roll', color='blue')

    # plt.title(f'Orientation Comparison ({axis.upper()} Axis) for {description} - {code}')
    plt.xlabel('Time in seconds')
    plt.ylabel(f'{axis.upper()} Orientation (degrees)')
    plt.legend()
    plt.grid()

    if save:
        filename = make_plot_name(
        "orientation_comparison",
        reference_label,
        comparison_label,
        code)
        _save_report_figure(f'{plots_path}/{f"{axis}_{filename}"}')
    else:
        print(f"Showing orientation comparison for {axis} axis with description: {description}_{code}")
        _apply_report_layout()
        plt.show()
        
    plt.close()  # Close the plot to free up memory

def plot_orientation(x_axis, orientation, code, description='unknown', axis='yaw', save=False,  plots_path='/opt/workspace/datasets/', label=None):
    plt.figure(figsize=figure_size('one_third'))
    x_axis = elapsed_time_in_seconds(x_axis) # convert the timestamps to elapsed time in seconds for better visualization

    orientation = np.array(orientation)
    orientation = np.unwrap(orientation, axis=0)
    orientation = np.degrees(orientation)

    plot_label = label if label is not None else description if description is not None else 'Odometry'
    if axis == 'yaw':
        plt.plot(x_axis, orientation[:, 2], label=f'{plot_label} Yaw', color='blue')
    elif axis == 'pitch':
        plt.plot(x_axis, orientation[:, 1], label=f'{plot_label} Pitch', color='blue')
    elif axis == 'roll':
        plt.plot(x_axis, orientation[:, 0], label=f'{plot_label} Roll', color='blue')

    # plt.title(f'Orientation plot of ({axis.upper()} Axis) for {description} - {code}')
    plt.xlabel('Time in seconds')
    plt.ylabel(f'{axis.upper()} Orientation (degrees)')
    plt.legend()
    plt.grid()

    if save:
        filename = make_plot_name(
        "orientation_plot_{axis}",
        None,        label,
        code)
        _save_report_figure(f'{plots_path}/{filename}')
    else:
        print(f"Showing orientation plot for {axis} axis with description: {description}_{code}")
        _apply_report_layout()
        plt.show()
        
    plt.close()  # Close the plot to free up memory

def plot_relative_pose_error_histogram(trans_errors, 
                   rot_errors,                 
                    code='unknown',
                    save=False,
                    show=False,
                    plots_path='/opt/workspace/datasets/',
                    odom_type='unknown',
                    comparison_label=None,
                    reference_label=None):
    
    plt.figure(figsize=figure_size('one_third'))
    plt.hist(trans_errors, bins=50, alpha=0.8)
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'
    # plt.title('Relative Pose Translation Error Distribution for ' + comp_label + ' vs ' + ref_label + ' - ' + code)
    plt.xlabel('Translation Error (m)')
    plt.ylabel('Frequency')
    plt.grid(True)

    if save:
        filename = make_plot_name('RPE_Translation_Error_Distribution',reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename))
    if show:
        plt.show()
    plt.close()
    
    # convert rot_errors from radians to degrees
    rot_errors = [np.degrees(err) for err in rot_errors]
    plt.figure(figsize=figure_size('one_third'))
    plt.hist(rot_errors, bins=50, color='green', alpha=0.7)
    # plt.title('Relative Pose Rotation Error Distribution for ' + comp_label + ' vs ' + ref_label + ' - ' + code)
    plt.xlabel('Rotation Error (deg)')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save:
        filename = make_plot_name('RPE_Rotation_Error_Distribution', reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename))
    if show:
        plt.show()
    plt.close()

def plot_relative_pose_translation_error(trans_error_list, 
                                        acceptable_max_error=1.0, 
                                        distance_list=None,
                                        code='unknown',
                                        save=False,
                                        show=True,
                                        plots_path='/opt/workspace/datasets/',
                                        odom_type='unknown',
                                        comparison_label=None,
                                        reference_label=None):
    """
    Plot translational RPE error over sliding windows/travel distance.
    
    trans_error_list: list or array of translational errors (m)
    distance_list: optional, cumulative distances corresponding to each window
    """
    plt.figure(figsize=figure_size('one_third'))
    
    if distance_list is None:
        x = range(len(trans_error_list))
        xlabel = 'Sliding window index'
    else:
        x = distance_list[:len(trans_error_list)]
        xlabel = 'Travel distance (m)'
    
    threshold = acceptable_max_error
    plt.scatter([x[i] for i, e in enumerate(trans_error_list) if e>threshold],
                [e for e in trans_error_list if e>threshold],
                color='red', label=f'Error > {acceptable_max_error} m')
    plt.fill_between(x, 0, acceptable_max_error, color='green', alpha=0.2, label='Acceptable range')
    plt.plot(x, trans_error_list, marker='x', linestyle='-', color='b', label='Translation Error', alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel('Translation Error (m)')
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'
    # plt.title('Translational RPE along trajectory of ' + comp_label + ' vs ' + ref_label + ' - ' + code)
    plt.grid(True)
    plt.legend()
    _apply_report_layout()
    if save:
        filename = make_plot_name('RPE_Translation_Error_Distribution', reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename))
    if show:    
        plt.show()

    plt.close()

def plot_relative_pose_rotation_error(rot_error_list, 
                                     acceptable_max_error=10.0, 
                                     distance_list=None,
                                     code='unknown',
                                     save=False,
                                     show=True,
                                     plots_path='/opt/workspace/datasets/',
                                     odom_type='unknown',
                                     comparison_label=None,
                                     reference_label=None):
    """
    Plot rotational RPE error over sliding windows/travel distance. 

    rot_error_list: list or array of rotational errors (degrees)
    acceptable_max_error: float, threshold for acceptable error in degrees
    distance_list: optional, cumulative distances corresponding to each window
    """
    plt.figure(figsize=figure_size('one_third'))
    if distance_list is None:
        x = range(len(rot_error_list))
        xlabel = 'Sliding window index'
    else:
        x = distance_list[:len(rot_error_list)]
        xlabel = 'Travel distance (m)'
    threshold = np.round(acceptable_max_error, 2)

    # convert rot_error_list from radians to degrees
    rot_error_list = [np.degrees(err) for err in rot_error_list]
    plt.scatter([x[i] for i, e in enumerate(rot_error_list) if e>threshold],
                [e for e in rot_error_list if e>threshold],
                color='red', label=f'Error > {threshold} deg')
    plt.fill_between(x, 0, threshold, color='green', alpha=0.2, label='Acceptable range')
    plt.plot(x, rot_error_list, marker='x', linestyle='-', color='b', label='Rotation Error', alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel('Rotation Error (deg)')
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'
    # plt.title('Rotational RPE along trajectory of ' + comp_label + ' vs ' + ref_label + ' - ' + code)
    plt.grid(True)
    plt.legend()
    _apply_report_layout()
    if save:
        filename = make_plot_name('RPE_Rotation_Error_Distribution', reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename))
    if show:    
        plt.show()
    plt.close()

def plot_outlier_removal(vicon_timestamp_, filtered_vicon_timestamp, vicon_yaw, odom_yaw, filtered_vicon_yaw, filtered_odom_yaw, idx_change, code, save=False, show=False,plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
    """
    Function to plot the yaw difference between VICON and odometry data
    :param vicon_timestamp_: The timestamps (in microseconds) of the VICON data
    :param filtered_vicon_timestamp: The filtered synced ground truth data (in microseconds)
    :param vicon_yaw: The VICON yaw data (in radians)
    :param odom_yaw: The odometry yaw data (in radians)
    :param filtered_vicon_yaw: The filtered VICON yaw data (in radians)
    :param filtered_odom_yaw: The filtered odometry yaw data (in radians)
    :param code: The code for the dataset(string)
    :param idx_change: The indices where the yaw changes significantly
    :return: None
    """
    fig, ax = plt.subplots(2, figsize=figure_size('one_third'), sharey=True)

    # setting data for plotting
    x_axis = elapsed_time_in_seconds(vicon_timestamp_)

    comp_label = comparison_label if comparison_label is not None else 'Odometry'
    comp_label = comparison_label if comparison_label is not None else 'comparison'
    ref_label = reference_label if reference_label is not None else 'reference'
    t_change = [x_axis[i] for i in idx_change]
    v_change = [np.deg2rad(vicon_yaw[i]) for i in idx_change]
    ax[0].scatter(t_change, np.degrees(v_change), color='green', label=f"{ref_label} Yaw Change", s=100, marker='o')
    ax[0].plot(x_axis, np.degrees(vicon_yaw), label=f"{ref_label} Yaw", color='blue')
    ax[0].plot(x_axis, np.degrees(odom_yaw), label=f"{comp_label} Yaw", color='red')
    _set_axis_title(ax[0], "Yaw data for dataset: " + code + " with outliers", fontsize=FONT_SIZE_LABELS)
    ax[0].grid()
    ax[0].legend(fontsize=FONT_SIZE_LEGEND)

    x_axis = elapsed_time_in_seconds(filtered_vicon_timestamp)

    ax[1].plot(x_axis, np.degrees(filtered_vicon_yaw), label=f"Filtered {ref_label} Yaw", color='blue')
    ax[1].plot(x_axis, np.degrees(filtered_odom_yaw), label=f"Filtered {comp_label} Yaw", color='red')
    _set_axis_title(ax[1], "Filtered Yaw data for dataset: " + code, fontsize=FONT_SIZE_LABELS)
    ax[1].grid()
    ax[1].legend(fontsize=FONT_SIZE_LEGEND)

    for a in ax.flat:
        a.tick_params(axis='x', labelsize=FONT_SIZE_TICKS)
        a.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)

    _apply_report_layout(fig=fig)
    fig.text(0.5, -0.01, 'Elapsed time in seconds', ha='center', fontsize=FONT_SIZE_LABELS)
    fig.text(-0.007, 0.5, 'Yaw angle in degrees', va='center', rotation='vertical', fontsize=FONT_SIZE_LABELS)
    if save:
        filename = make_plot_name('Yaw_Comparison', reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename), fig=fig)
    if show:
        plt.show()
    else:
        plt.close(fig) 

def plot_error_histogram(vicon_data, odom_data, code, axis='yaw', odom_type='unknown', mean_center=False, offset=None, save=False, show=False,plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):

    """
    Plot and save the histogram of the error between the filtered vicon and odometry data
    :param filtered_vicon_data: filtered vicon data (in radians)
    :param filtered_odom_data: filtered odometry data (in radians)
    :param code: code for the dataset

    :param mean_center: if True, center the data around the mean of the data ie, subtract the mean from the data
    :param offset: if not None, shift the data by the offset value (in degrees)
    :return: None
    """
    if axis not in ['yaw', 'pitch', 'roll']:
        raise ValueError("Invalid axis, must be one of 'yaw', 'pitch', 'roll'")
    
    if axis == 'yaw':
        vicon_data = vicon_data[:, 2]
        odom_data = odom_data[:, 2]
    elif axis == 'pitch':
        vicon_data = vicon_data[:, 1]
        odom_data = odom_data[:, 1]
    elif axis == 'roll':
        vicon_data = vicon_data[:, 0]
        odom_data = odom_data[:, 0]

    error_data =  np.degrees(fa(vicon_data, odom_data))

    # shift the error data to be centered around 0
    if mean_center:
        error_data = error_data - np.mean(error_data)

    # Sift the error to a fixed offset
    if offset is not None:
        error_data = error_data - offset

    # find the standard deviation of the error data
    mean_error = np.mean(error_data)
    std_error = np.std(error_data)

    plt.figure(figsize=FIGSIZE_HALF)
    plt.hist(error_data, bins=100, color='royalblue', alpha=1.0, edgecolor='black')
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=4, label='Mean')
    plt.axvline(mean_error + std_error, color='green', linestyle='dashed', linewidth=3, label='1 Sigma')
    plt.axvline(mean_error - std_error, color='green', linestyle='dashed', linewidth=3)
    plt.axvline(mean_error + 2*std_error, color='orange', linestyle='dashed', linewidth=2, label='2 Sigma')
    plt.axvline(mean_error - 2*std_error, color='orange', linestyle='dashed', linewidth=2)
    plt.axvline(mean_error + 3*std_error, color='purple', linestyle='dashed', linewidth=1, label='3 Sigma')
    plt.axvline(mean_error - 3*std_error, color='purple', linestyle='dashed', linewidth=1)
    comp_label = comparison_label if comparison_label is not None else odom_type
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'
    # plt.title("Histogram of "+axis+" difference(Error) between "+comp_label+" and "+ref_label+" for dataset: " + code, fontsize=20)
    plt.xlabel(f'{axis} difference (in degrees)', fontsize=FONT_SIZE_LABELS)
    plt.ylabel("Number of datapoints", fontsize=FONT_SIZE_LABELS)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    if save:
        filename = make_plot_name('Histogram_of_Error_Distribution', reference_label, comparison_label, code)
        _save_report_figure(os.path.join(plots_path, filename))
    if show:
        plt.show()
        
    plt.close()  # Close the plot to free up memory

def plot_trajectory_with_heading(odom_position, vicon_position, odom_yaw=None, vicon_yaw=None, code='unknown',odom_type='unknown',scale=3, width=TRAJECTORY_HEADING_VECTOR_WIDTH, start_point=500, step=100, heading=True, save=False, show=False,plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
    """
    Plot the trajectory of the odometry and ground truth data with heading vectors.
    :param odom_position: The odometry position data as a 2D/3D array of shape (n, 2 or 3). only x and y coordinates are used
    :param vicon_position: The ground truth position data as a 2D/3D array of shape (n, 2 or 3). only x and y coordinates are used
    :param odom_yaw: The odometry yaw angles as a 1D array of shape (n,).
    :param vicon_yaw: The ground truth yaw angles as a 1D array of shape (n,).
    :param code: The code to be used in the plot title and filename.
    :return: None
    """

    if not heading:
        odom_yaw = np.zeros(odom_position.shape)
        vicon_yaw = np.zeros(vicon_position.shape)
    elif odom_yaw is None or vicon_yaw is None:
        raise ValueError("The odometry and ground truth yaw angles must be provided if heading is True")
    #check if the lengths of the positions and yaw angles are the same
    if not (len(odom_position) == len(odom_yaw) == len(vicon_position) == len(vicon_yaw)):
        raise ValueError("The lengths of the positions and yaw angles must be the same")

    fig, ax = plt.subplots(figsize=TRAJ_FIGSIZE)

    ax.set_aspect('equal', adjustable='box')  # ensures 1:1 scaling 
    ax.autoscale(enable=True)                     # auto-adjust limits

    scale = scale # the higher the value, the shorter the arrows
    width = width # the higher the value, the thinner the arrows
    start_point = start_point # the starting point of the arrows (the first n points are not shown) (there is a lot of orientation data at the beginning of the trajectory)
    step = step # the step size for the arrows (the higher the value, the less arrows are shown) (to keep the plot clean)

    if heading:
        # Use the yaw angles to create a 2D heading vectors(hv)
        hv_odom = np.array([np.cos(odom_yaw), np.sin(odom_yaw)]).T
        hv_vicon = np.array([np.cos(vicon_yaw), np.sin(vicon_yaw)]).T

    comp_label = comparison_label if comparison_label is not None else odom_type
    ref_label = reference_label if reference_label is not None else 'Ground truth (VICON)'
    comp_color = _trajectory_role_color(comp_label, 'comparison')
    ref_color = _trajectory_role_color(ref_label, 'reference')
    ax.plot(odom_position[:,0], odom_position[:,1], label=f'{comp_label}', color=comp_color, alpha=TRAJECTORY_COMPARISON_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.plot(vicon_position[:,0], vicon_position[:,1], label=ref_label, color=ref_color, alpha=TRAJECTORY_REFERENCE_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)

    label_added = False
    # plot the heading vectors for odometry

    if heading:
        cnt = 0
        for x, y, p, q in zip(odom_position[:, 0], odom_position[:, 1], hv_odom[:, 0], hv_odom[:, 1]):
            if cnt % step == 0 and cnt > start_point:
                ax.quiver(x, y, p, q, angles='xy', scale_units='xy', scale=scale, alpha=TRAJECTORY_COMPARISON_HEADING_ALPHA, color=comp_color, width=width, label=f"{comp_label} heading vectors" if (TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND and not label_added) else None)
                label_added = True
            else:
                pass
            cnt += 1


        label_added = False
        cnt = 0
        for x, y, p, q in zip(vicon_position[:, 0], vicon_position[:, 1], hv_vicon[:, 0], hv_vicon[:, 1]):
            if cnt % step == 0 and cnt > start_point:
                ax.quiver(x, y, p, q, angles='xy', scale_units='xy', scale=scale, alpha=TRAJECTORY_REFERENCE_HEADING_ALPHA, color=ref_color, width=width, label=f"{ref_label} heading vectors" if (TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND and not label_added) else None)
                label_added = True
            else:
                pass
            cnt += 1
    else:
        print("Heading vecotor plotting is disabled, enable it by setting heading=True")

    _set_trajectory_view(ax, [odom_position, vicon_position])

    ax.scatter(odom_position[0,0], odom_position[0,1], color='green', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE)
    ax.scatter(odom_position[-1,0], odom_position[-1,1], color='black', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE)
    ax.scatter(vicon_position[0,0], vicon_position[0,1], color='green', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE, label="start point")
    ax.scatter(vicon_position[-1,0], vicon_position[-1,1], color='black', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE, label="end point")
    
    # plt.title("Trajectory with heading comparison between "+comp_label+" and "+ref_label+" for dataset: " + code)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.grid()
    _set_trajectory_legend(ax, fig)
    if TRAJECTORY_LEGEND_LOCATION != 'outside_bottom':
        _apply_report_layout(pad=0.2)
    if save:
        if heading:
            filename = make_plot_name('Trajectory_With_Heading', reference_label, comparison_label, code)
            _save_report_figure(os.path.join(plots_path, filename), fig=fig, bbox_inches=TRAJECTORY_SAVE_BBOX_INCHES)
        else:
            filename = make_plot_name('Trajectory', reference_label, comparison_label, code)
            _save_report_figure(os.path.join(plots_path, filename), fig=fig, bbox_inches=TRAJECTORY_SAVE_BBOX_INCHES)
    if show:
        plt.show()
        
    plt.close(fig)

def plot_position_error_over_time(odom_timestamps, odom_position, vicon_position, code, odom_type=None, save=False, show=False, plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
    """
    Plot the position error between the odometry and ground truth data over time.
    Generates separate plots for x, y, and z position error.

    :param odom_timestamps: The timestamps of the odometry data (in microseconds)
    :param odom_position: The position data of the odometry (Nx3 array)
    :param vicon_position: The position data of the ground truth (Nx3 array)
    :param code: The code for the dataset
    :return: None
    """
    x_axis = elapsed_time_in_seconds(odom_timestamps)
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type is not None else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'

    filename_base = make_plot_name('Position_Error', reference_label, comparison_label, code)
    axis_specs = [
        ('x', 0, 'X', 'red', f'x_{filename_base}'),
        ('y', 1, 'Y', 'red', f'y_{filename_base}'),
        ('z', 2, 'Z', 'red', f'z_{filename_base}'),
    ]

    for axis_name, axis_idx, axis_label, color, filename in axis_specs:
        axis_error = odom_position[:, axis_idx] - vicon_position[:, axis_idx]

        plt.figure(figsize=figure_size('one_third'))
        plt.plot(x_axis, axis_error, label=f'{axis_label} position Error ({comp_label} - {ref_label})', color=color)
        # plt.title(f'{axis_label} position Error over Time for ' + comp_label + ' - ' + code)
        plt.xlabel('Time (s)')
        plt.ylabel(f'{axis_label} position Error (m)')
        plt.grid()
        plt.legend()

        if save:
            _save_report_figure(f'{plots_path}/{filename}')
        if show:
            _apply_report_layout()
            plt.show()

        plt.close()  # Close the plot to free up memory


# def plot_z_error_over_time(odom_timestamps, odom_position, vicon_position, code, odom_type=None, save=False, show=False, plots_path='/opt/workspace/datasets/', comparison_label=None, reference_label=None):
#     return plot_position_error_over_time(
#         odom_timestamps,
#         odom_position,
#         vicon_position,
#         code,
#         odom_type=odom_type,
#         save=save,
#         show=show,
#         plots_path=plots_path,
#         comparison_label=comparison_label,
#         reference_label=reference_label,
#     )

def plot_trajectory_heatmap_time(odom_position, vicon_position,
                                  odom_timestamps, vicon_timestamps,
                                  odom_yaw=None, vicon_yaw=None,
                                  code='unknown', odom_type='unknown',
                                  heading=True,
                                  scale=3, width=TRAJECTORY_HEADING_VECTOR_WIDTH, start_point=500, step=100,
                                  save=False, show=False,
                                  plots_path='/opt/workspace/datasets/',
                                  comparison_label=None, reference_label=None):
    """
    Plot the trajectory of the odometry and ground truth data as a heatmap with respect to elapsed time.
    param odom_position: The position data of the odometry (Nx3 array)
    param vicon_position: The position data of the ground truth (Nx3 array)
    param odom_timestamps: The timestamps of the odometry data (in microseconds)
    param vicon_timestamps: The timestamps of the ground truth data (in microseconds)
    param odom_yaw: The yaw angles of the odometry data (in radians)
    param vicon_yaw: The yaw angles of the ground truth data (in radians)
    param code: The code for the dataset
    param heading: Whether to plot the heading vectors or not
    param scale: The scale of the heading vectors (the higher the value, the shorter the arrows)
    param width: The width of the heading vectors (the higher the value, the thinner the arrows)
    param start_point: The starting point of the heading vectors (the first n points are not shown) (there is a lot of orientation data at the beginning of the trajectory)
    param step: The step size for the heading vectors (the higher the value, the less arrows are shown) (to keep the plot clean)    
    param save: Whether to save the plot or not
    param show: Whether to show the plot or not
    param plots_path: The path to save the plot if save is True
    param comparison_label: The label for the odometry data to be used in the plot title and legend
    param reference_label: The label for the ground truth data to be used in the plot title and legend
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if not heading:
        odom_yaw = np.zeros(len(odom_position))
        vicon_yaw = np.zeros(len(vicon_position))
    elif odom_yaw is None or vicon_yaw is None:
        raise ValueError("The odometry and ground truth yaw angles must be provided if heading is True")

    if not (len(odom_position) == len(odom_yaw)):
        raise ValueError("The lengths of the odometry positions and yaw angles must be the same")
    if not (len(vicon_position) == len(vicon_yaw)):
        raise ValueError("The lengths of the ground truth positions and yaw angles must be the same")

    odom_t  = np.array(elapsed_time_in_seconds(odom_timestamps))
    vicon_t = np.array(elapsed_time_in_seconds(vicon_timestamps))

    odom_t_norm  = (odom_t  - odom_t.min())  / (odom_t.max()  - odom_t.min())
    vicon_t_norm = (vicon_t - vicon_t.min()) / (vicon_t.max() - vicon_t.min())

    cmap = plt.cm.plasma

    comp_label = comparison_label if comparison_label is not None else odom_type
    comp_label = comparison_label if comparison_label is not None else (odom_type if odom_type and odom_type != 'unknown' else 'comparison')
    ref_label = reference_label if reference_label is not None else 'reference'
    fig, axes = plt.subplots(1, 2, figsize=TRAJ_COMPARISON_FIGSIZE, constrained_layout=True)
    _set_figure_title(fig, "Trajectory heatmap (elapsed time) comparison between " + comp_label + " and " + ref_label + " for dataset: " + code)

    for ax, pos, yaw, t_norm, title in zip(
        axes,
        [odom_position,  vicon_position],
        [odom_yaw,       vicon_yaw],
        [odom_t_norm,    vicon_t_norm],
        [comp_label,      ref_label]
    ):
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale(enable=True)

        # plot trajectory as heatmap
        points   = pos[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1), linewidth=TRAJECTORY_LINE_WIDTH, alpha=0.85)
        lc.set_array(t_norm[:-1])
        ax.add_collection(lc)

        if heading:
            hv = np.array([np.cos(yaw), np.sin(yaw)]).T

            label_added = False
            cnt = 0
            for x, y, p, q, t in zip(pos[:, 0], pos[:, 1], hv[:, 0], hv[:, 1], t_norm):
                if cnt % step == 0 and cnt > start_point:
                    ax.quiver(x, y, p, q, angles='xy', scale_units='xy', scale=scale, alpha=0.6, color=cmap(t), width=width, label="heading vectors" if not label_added else None)
                    label_added = True
                else:
                    pass
                cnt += 1
        else:
            print("heading vectors plotting is disabled, enable it by setting heading=True")

        ax.scatter(pos[0,  0], pos[0,  1], color='green', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE, zorder=5, label="start point")
        ax.scatter(pos[-1, 0], pos[-1, 1], color='black', marker='o', s=TRAJECTORY_ENDPOINT_MARKER_SIZE, zorder=5, label="end point")

        _set_trajectory_view(ax, [pos])

        _set_axis_title(ax, title)
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        _set_trajectory_legend(ax, fig, include_heading_vectors=False)
        ax.grid()

    # shared colorbar
    t_global_max = max(odom_t.max(), vicon_t.max())
    norm = Normalize(vmin=0, vmax=t_global_max)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("Elapsed time (s)")

    if save:
        if heading:
            filename = make_plot_name('TimeHeatmap_Trajectory_With_Heading', reference_label, comparison_label, code)
            _save_report_figure(f'{plots_path}/{filename}')
        else:
            filename = make_plot_name('TimeHeatmap_Trajectory', reference_label, comparison_label, code)
            _save_report_figure(f'{plots_path}/{filename}')
    if show:
        plt.show()

    plt.close(fig)


def plot_timestamp_synchronization(
    original_timestamps,
    synced_timestamps,
    code,
    stream_name='comparison',
    save=False,
    show=False,
    plots_path='/opt/workspace/datasets/'
):
    """
    Visualize timestamps before and after synchronization.

    Parameters
    ----------
    original_timestamps : np.ndarray
        Timestamps before synchronization.
    synced_timestamps : np.ndarray
        Timestamps after synchronization.
    code : str
        Dataset code.
    stream_name : str
        Name of the stream ('comparison' or 'reference').
    """

    original = np.asarray(original_timestamps)
    synced = np.asarray(synced_timestamps)

    removed = np.setdiff1d(original, synced)

    plt.figure(figsize=figure_size('one_third'))

    # Original timestamps
    plt.scatter(
        elapsed_time_in_seconds(original),
        np.zeros(len(original)),
        marker='|',
        s=100,
        label=f'Original ({len(original)})'
    )

    # Synced timestamps
    plt.scatter(
        elapsed_time_in_seconds(synced),
        np.ones(len(synced)),
        marker='|',
        s=100,
        label=f'Synced ({len(synced)})'
    )

    # Removed timestamps
    if len(removed) > 0:
        plt.scatter(
            elapsed_time_in_seconds(removed),
            np.full(len(removed), 0.5),
            marker='x',
            s=25,
            label=f'Removed ({len(removed)})'
        )

    plt.yticks(
        [0, 0.5, 1],
        ['Original', 'Removed', 'Synced']
    )

    plt.xlabel('Time (s)')
    plt.ylabel('Timestamp State')
    # plt.title(f'Timestamp Synchronization - {stream_name} - {code}')
    plt.grid()
    plt.legend()

    filename = f'timestamp_sync_{stream_name}_{code}.png'

    if save:
        _save_report_figure(f'{plots_path}/{filename}')

    # print("save path", f'{plots_path}/{filename}')

    if show:
        plt.show()

    plt.close()

def plot_timestamp_comparison(
    comparison_timestamps,
    reference_timestamps,
    code,
    matching_threshold_us=10000,
    save=False,
    show=False,
    plots_path='/opt/workspace/datasets/',
    comparison_label=None,
    reference_label=None
):
    """
    Plot timestamps from two streams before synchronization and highlight
    reference timestamps that do not have a matching comparison timestamp.

    Parameters
    ----------
    comparison_timestamps : np.ndarray
        Timestamps from the stream being evaluated.
    reference_timestamps : np.ndarray
        Timestamps from the reference stream.
    code : str
        Dataset code.
    matching_threshold_us : float
        Maximum allowed timestamp difference (in microseconds) to be
        considered a match.
    save : bool
        Whether to save the plot.
    show : bool
        Whether to display the plot.
    plots_path : str
        Directory to save the plot.
    comparison_label : str
        Label for the comparison stream.
    reference_label : str
        Label for the reference stream.
    """

    comparison_label = comparison_label or "Comparison"
    reference_label = reference_label or "Reference"

    comparison = np.asarray(comparison_timestamps)
    reference = np.asarray(reference_timestamps)

    # ------------------------------------------------------------------
    # Find reference timestamps without a matching comparison timestamp
    # ------------------------------------------------------------------
    missing_reference = []

    if len(comparison) > 0:
        for ts in reference:
            nearest_idx = np.argmin(np.abs(comparison - ts))

            if np.abs(comparison[nearest_idx] - ts) > matching_threshold_us:
                missing_reference.append(ts)
    else:
        missing_reference = reference.copy()

    missing_reference = np.asarray(missing_reference)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=figure_size('one_third'))

    # Reference timestamps
    plt.scatter(
        elapsed_time_in_seconds(reference),
        np.ones(len(reference)),
        marker='|',
        s=100,
        label=f'{reference_label} ({len(reference)})'
    )

    # Comparison timestamps
    plt.scatter(
        elapsed_time_in_seconds(comparison),
        np.zeros(len(comparison)),
        marker='|',
        s=100,
        label=f'{comparison_label} ({len(comparison)})'
    )

    # Missing timestamps
    if len(missing_reference) > 0:
        plt.scatter(
            elapsed_time_in_seconds(missing_reference),
            np.full(len(missing_reference), 0.5),
            marker='x',
            s=30,
            label=f'No matching {comparison_label} ({len(missing_reference)})'
        )

    plt.yticks(
        [0, 0.5, 1],
        [comparison_label, 'Missing', reference_label]
    )

    plt.xlabel('Time (s)')
    plt.ylabel('Stream')

    # plt.title(f'Timestamp Comparison\n' f'{comparison_label} vs {reference_label} - {code}')    

    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    # print(f'{reference_label} samples : {len(reference)}')
    # print(f'{comparison_label} samples: {len(comparison)}')
    # print(f'Missing matches      : {len(missing_reference)}')

    if len(reference) > 0:
        print(
            f'Missing percentage   : '
            f'{100.0 * len(missing_reference) / len(reference):.2f}%'
        )

    # ------------------------------------------------------------------
    # Save / Show
    # ------------------------------------------------------------------
    filename = (
        f'timestamp_comparison_'
        f'{comparison_label.lower().replace(" ", "_")}_'
        f'{code}.png'
    )

    if save:
        _save_report_figure(f'{plots_path}/{filename}')

    # print("save path", f'{plots_path}/{filename}')

    if show:
        plt.show()

    plt.close()

    return missing_reference

def get_wheel_name_from_id(wheel_id):
    if wheel_id == 0:
        return "Front Left"
    if wheel_id == 1:
        return "Rear Left"
    if wheel_id == 2:
        return "Rear Right"
    if wheel_id == 3:
        return "Front Right"
    raise ValueError(f"Invalid wheel_id: {wheel_id}")


def get_wheel_id_from_contact_group_id(group_id, spokes_per_wheel=5):
    group_id = int(group_id)
    wheel_id = group_id // spokes_per_wheel

    if wheel_id < 0 or wheel_id >= 4:
        raise ValueError(f"Invalid contact group id: {group_id}")

    return wheel_id


def load_contact_points_from_msgpack(msgpack_path, stream_name):
    """
    Load contact debug stream from a msgpack file.

    Expected stream sample format:
        sample["time"]["microseconds"]
        sample["points"] -> list of contact points
        point["groupId"]

    Returns
    -------
    dict:
        {
            timestamp_us: [group_id_0, group_id_1, ...]
        }
    """

    import msgpack
    import pandas as pd

    data = msgpack.load(open(msgpack_path, "rb"))

    if stream_name not in data:
        raise KeyError(
            f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}"
        )

    dataframe = pd.DataFrame(data[stream_name])

    contact_data = {}

    for time, points in zip(dataframe["time"], dataframe["points"]):
        timestamp_us = time["microseconds"]
        contact_data[timestamp_us] = []

        for point in points:
            contact_data[timestamp_us].append(int(point["groupId"]))

    return contact_data


def group_contact_points_by_wheel(contact_points, num_wheels=4, spokes_per_wheel=5):
    """
    Convert a flat contact list into contacts grouped by wheel.

    Input:
        [0, 5, 10, 15]

    Output:
        [
            [0],    # front left
            [5],    # rear left
            [10],   # rear right
            [15],   # front right
        ]
    """

    grouped = [[] for _ in range(num_wheels)]

    for group_id in contact_points:
        wheel_id = get_wheel_id_from_contact_group_id(
            group_id,
            spokes_per_wheel=spokes_per_wheel,
        )
        grouped[wheel_id].append(int(group_id))

    return grouped


def build_contact_dict_by_wheel(contact_data, num_wheels=4, spokes_per_wheel=5):
    """
    Convert:
        {timestamp: [group_ids]}

    into:
        {timestamp: [[wheel0_contacts], [wheel1_contacts], ...]}
    """

    grouped_contact_data = {}

    for timestamp, contact_points in contact_data.items():
        grouped_contact_data[timestamp] = group_contact_points_by_wheel(
            contact_points,
            num_wheels=num_wheels,
            spokes_per_wheel=spokes_per_wheel,
        )

    return grouped_contact_data


def sync_contact_debug_streams(lf_contact_data, md_contact_data, max_time_diff_us=10000):
    """
    Sync LF and MD contact debug streams by nearest timestamp.

    Returns
    -------
    list of tuple:
        [(lf_timestamp, md_timestamp), ...]
    """

    lf_times = sorted(lf_contact_data.keys())
    md_times = sorted(md_contact_data.keys())

    synced_pairs = []

    md_idx = 0

    for lf_t in lf_times:
        while md_idx < len(md_times) and md_times[md_idx] < lf_t - max_time_diff_us:
            md_idx += 1

        candidates = []

        if md_idx > 0:
            candidates.append(md_times[md_idx - 1])

        if md_idx < len(md_times):
            candidates.append(md_times[md_idx])

        if not candidates:
            continue

        best_md_t = min(candidates, key=lambda t: abs(t - lf_t))

        if abs(best_md_t - lf_t) <= max_time_diff_us:
            synced_pairs.append((lf_t, best_md_t))

    return synced_pairs


def calculate_contact_selection_metrics(
    lf_contact_data,
    md_contact_data,
    max_time_diff_us=10000,
    num_wheels=4,
    spokes_per_wheel=5,
):
    """
    Compare Mahalanobis-selected contacts against lowest-foot contacts.

    Important:
    LF is only a geometric proxy, not true contact ground truth.

    Metrics:
        exact_match_rate:
            LF contact set == MD contact set

        recall:
            fraction of LF contacts also selected by MD

        precision:
            fraction of MD contacts that are also LF contacts

        jaccard:
            TP / (TP + FP + FN)

        contact_selection_loss:
            Lower is better.
            Penalizes missed LF contacts, extra MD contacts, and no MD contact.
    """

    lf_grouped = build_contact_dict_by_wheel(
        lf_contact_data,
        num_wheels=num_wheels,
        spokes_per_wheel=spokes_per_wheel,
    )

    md_grouped = build_contact_dict_by_wheel(
        md_contact_data,
        num_wheels=num_wheels,
        spokes_per_wheel=spokes_per_wheel,
    )

    synced_pairs = sync_contact_debug_streams(
        lf_grouped,
        md_grouped,
        max_time_diff_us=max_time_diff_us,
    )

    if len(synced_pairs) == 0:
        raise ValueError("No synced LF/MD contact debug samples found.")

    per_wheel = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_exact = 0
    total_samples = 0
    total_loss = 0.0
    total_md_contacts = 0
    total_lf_contacts = 0

    for wheel_id in range(num_wheels):
        tp = 0
        fp = 0
        fn = 0
        exact = 0
        samples = 0
        loss_sum = 0.0
        md_contacts_count = 0
        lf_contacts_count = 0
        no_md_contact_count = 0
        extra_contact_count = 0

        for lf_t, md_t in synced_pairs:
            lf_set = set(lf_grouped[lf_t][wheel_id])
            md_set = set(md_grouped[md_t][wheel_id])

            tp_i = len(lf_set & md_set)
            fp_i = len(md_set - lf_set)
            fn_i = len(lf_set - md_set)

            tp += tp_i
            fp += fp_i
            fn += fn_i

            md_contacts_count += len(md_set)
            lf_contacts_count += len(lf_set)

            if lf_set == md_set:
                exact += 1

            if len(md_set) == 0:
                no_md_contact_count += 1

            extra_contacts = max(0, len(md_set) - len(lf_set))
            extra_contact_count += extra_contacts

            # Loss:
            # - missed LF contact is bad
            # - extra MD contact is bad
            # - selecting no contact is also bad
            loss = 0.0
            loss += 1.0 * fn_i
            loss += 0.5 * fp_i

            if len(md_set) == 0:
                loss += 1.0

            loss_sum += loss
            samples += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        exact_match_rate = exact / samples if samples > 0 else 0.0
        mean_md_contacts = md_contacts_count / samples if samples > 0 else 0.0
        mean_lf_contacts = lf_contacts_count / samples if samples > 0 else 0.0
        contact_selection_loss = loss_sum / samples if samples > 0 else float("nan")

        per_wheel[wheel_id] = {
            "wheel_name": get_wheel_name_from_id(wheel_id),
            "samples": int(samples),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "jaccard": float(jaccard),
            "exact_match_rate": float(exact_match_rate),
            "mean_lf_contacts": float(mean_lf_contacts),
            "mean_md_contacts": float(mean_md_contacts),
            "no_md_contact_rate": float(no_md_contact_count / samples) if samples > 0 else 0.0,
            "extra_contact_rate": float(extra_contact_count / samples) if samples > 0 else 0.0,
            "contact_selection_loss": float(contact_selection_loss),
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_exact += exact
        total_samples += samples
        total_loss += loss_sum
        total_md_contacts += md_contacts_count
        total_lf_contacts += lf_contacts_count

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    overall_exact_match_rate = total_exact / total_samples if total_samples > 0 else 0.0
    overall_loss = total_loss / total_samples if total_samples > 0 else float("nan")

    return {
        "synced_samples": int(len(synced_pairs)),
        "per_wheel": per_wheel,
        "overall": {
            "samples": int(total_samples),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "jaccard": float(overall_jaccard),
            "exact_match_rate": float(overall_exact_match_rate),
            "mean_lf_contacts": float(total_lf_contacts / total_samples),
            "mean_md_contacts": float(total_md_contacts / total_samples),
            "contact_selection_loss": float(overall_loss),
        },
        "synced_pairs": synced_pairs,
    }


def plot_contact_selection_comparison(
    lf_contact_data,
    md_contact_data,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    max_time_diff_us=10000,
    num_wheels=4,
    spokes_per_wheel=5,
):
    """
    Plot LF proxy contacts and MD selected contacts over time.

    Saves:
        Contact_Selection_LF_vs_MD_<code>.png
    """

    lf_grouped = build_contact_dict_by_wheel(
        lf_contact_data,
        num_wheels=num_wheels,
        spokes_per_wheel=spokes_per_wheel,
    )

    md_grouped = build_contact_dict_by_wheel(
        md_contact_data,
        num_wheels=num_wheels,
        spokes_per_wheel=spokes_per_wheel,
    )

    synced_pairs = sync_contact_debug_streams(
        lf_grouped,
        md_grouped,
        max_time_diff_us=max_time_diff_us,
    )

    if len(synced_pairs) == 0:
        raise ValueError("No synced LF/MD contact debug samples found for plotting.")

    lf_times = [pair[0] for pair in synced_pairs]
    elapsed_times = elapsed_time_in_seconds(lf_times)

    fig, ax = plt.subplots(
        num_wheels,
        1,
        figsize=FIGSIZE_STACKED,
        sharex=True,
        gridspec_kw={"hspace": 0.25},
    )

    _set_figure_title(fig, f"Contact Selection Comparison: LF proxy vs MD - {code}", fontsize=FONT_SIZE_LABELS)

    for wheel_id in range(num_wheels):
        x_lf, y_lf = [], []
        x_md, y_md = [], []

        for idx, (lf_t, md_t) in enumerate(synced_pairs):
            elapsed_t = elapsed_times[idx]

            for spoke in lf_grouped[lf_t][wheel_id]:
                x_lf.append(elapsed_t)
                y_lf.append(spoke)

            for spoke in md_grouped[md_t][wheel_id]:
                x_md.append(elapsed_t)
                y_md.append(spoke)

        ax[wheel_id].scatter(
            x_md,
            y_md,
            label="Mahalanobis selected",
            marker="x",
            s=40,
            color="red",
            alpha=0.5,
        )

        ax[wheel_id].scatter(
            x_lf,
            y_lf,
            label="Lowest-foot proxy",
            marker="o",
            s=10,
            color="blue",
            alpha=0.6,
        )

        ax[wheel_id].set_ylabel(get_wheel_name_from_id(wheel_id))
        ax[wheel_id].set_yticks(
            range(
                wheel_id * spokes_per_wheel,
                (wheel_id + 1) * spokes_per_wheel,
            )
        )
        ax[wheel_id].grid(axis="y")

    ax[-1].set_xlabel("Elapsed time [s]")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.96),
        ncol=2,
    )

    filename = f"Contact_Selection_LF_vs_MD_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, filename))

    if show:
        plt.show()

    plt.close(fig)


def plot_contact_selection_metrics_bar(
    contact_metrics,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
):
    """
    Plot contact precision/recall/Jaccard per wheel.

    Saves:
        Contact_Selection_Metrics_<code>.png
    """

    wheel_names = []
    precision = []
    recall = []
    jaccard = []
    exact = []

    for wheel_id in range(4):
        data = contact_metrics["per_wheel"][wheel_id]
        wheel_names.append(data["wheel_name"])
        precision.append(100.0 * data["precision"])
        recall.append(100.0 * data["recall"])
        jaccard.append(100.0 * data["jaccard"])
        exact.append(100.0 * data["exact_match_rate"])

    x = np.arange(len(wheel_names))
    width = 0.2

    plt.figure(figsize=figure_size("one_third"))
    plt.bar(x - 1.5 * width, precision, width, label="Precision")
    plt.bar(x - 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 0.5 * width, jaccard, width, label="Jaccard")
    plt.bar(x + 1.5 * width, exact, width, label="Exact set match")

    plt.xticks(x, wheel_names)
    plt.ylabel("Score [%]")
    plt.ylim(0, 100)
    # plt.title(f"Contact Selection Metrics - {code}")
    plt.grid(axis="y")
    plt.legend()
    _apply_report_layout()

    filename = f"Contact_Selection_Metrics_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, filename))

    if show:
        plt.show()

    plt.close()
    
    
def load_imu_bias_from_msgpack(
    msgpack_path,
    stream_name="coyote3_contact_velocity_odometry_estimator.imu_bias_estimated",
):
    """
    Load IMU bias estimate from a msgpack file.

    Expected stream sample format:
        sample["time"]["microseconds"]
        sample["gyro"]["data"]
        sample["acc"]["data"] or sample["accelerometer"]["data"]

    Returns
    -------
    timestamps : np.ndarray
        Timestamp array in microseconds.

    gyro_bias : np.ndarray
        Nx3 gyro bias array in rad/s.

    acc_bias : np.ndarray
        Nx3 accelerometer bias array in m/s^2.
    """

    data = msgpack.load(open(msgpack_path, "rb"))

    if stream_name not in data:
        raise KeyError(
            f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}"
        )

    dataframe = pd.DataFrame(data[stream_name])

    timestamps = []
    gyro_bias = []
    acc_bias = []

    for _, row in dataframe.iterrows():
        timestamps.append(row["time"]["microseconds"])
        gyro_bias.append(row["gyro"]["data"])

        if "acc" in dataframe.columns:
            acc_bias.append(row["acc"]["data"])
        elif "accelerometer" in dataframe.columns:
            acc_bias.append(row["accelerometer"]["data"])
        else:
            raise KeyError(
                "Could not find accelerometer bias field. "
                "Expected column 'acc' or 'accelerometer'."
            )

    return (
        np.asarray(timestamps),
        np.asarray(gyro_bias),
        np.asarray(acc_bias),
    )
def load_debug_measurement_imu_from_msgpack(
    msgpack_path,
    stream_name="coyote3_contact_velocity_odometry_estimator.debug_measurement_imu",
):
    """
    Load debug IMU measurement stream from a msgpack file.

    Expected stream sample format:
        sample["time"]["microseconds"]
        sample["acc"]["data"]
        sample["gyro"]["data"]
        sample["mag"]["data"]

    Returns
    -------
    timestamps : np.ndarray
        Timestamp array in microseconds.

    acc : np.ndarray
        Nx3 accelerometer measurement array.

    gyro : np.ndarray
        Nx3 gyroscope measurement array.

    mag : np.ndarray
        Nx3 magnetometer measurement array.
    """

    data = msgpack.load(open(msgpack_path, "rb"))

    if stream_name not in data:
        raise KeyError(
            f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}\n"
            f"Available streams:\n{list(data.keys())}"
        )

    dataframe = pd.DataFrame(data[stream_name])
    
    # print(f"Loaded {len(dataframe)} samples from stream '{stream_name}' in msgpack file: {msgpack_path}")
    # print(f"Available columns in the dataframe: {list(dataframe.columns)}")

    timestamps = []
    acc = []
    gyro = []
    mag = []

    for _, row in dataframe.iterrows():
        timestamps.append(row["time"]["microseconds"])

        acc.append(row["acc"]["data"])
        gyro.append(row["gyro"]["data"])
        mag.append(row["mag"]["data"])

    return (
        np.asarray(timestamps),
        np.asarray(gyro),
        np.asarray(acc),
    )

def plot_imu_and_bias_evolution(
    timestamps,
    gyro_bias,
    acc_bias,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    start=None,
    end=None,
    comparison_label="contact_velocity_odometry",
    imu_or_bias="body_rotated_imu"
):
    """
    Plot gyro and accelerometer bias evolution.

    Saves:
        IMU_Bias_Evolution_<comparison_label>_<code>.png
    """

    timestamps = np.asarray(timestamps)
    gyro_bias = np.asarray(gyro_bias)
    acc_bias = np.asarray(acc_bias)

    if timestamps.size == 0:
        raise ValueError("Empty timestamp data. Cannot plot IMU bias evolution.")

    if gyro_bias.ndim != 2 or gyro_bias.shape[1] != 3:
        raise ValueError(f"Expected gyro_bias shape Nx3, got {gyro_bias.shape}")

    if acc_bias.ndim != 2 or acc_bias.shape[1] != 3:
        raise ValueError(f"Expected acc_bias shape Nx3, got {acc_bias.shape}")

    if start is None:
        start = 0

    if end is None:
        end = len(timestamps)

    start = max(0, int(start))
    end = min(len(timestamps), int(end))

    time_s = elapsed_time_in_seconds(timestamps)

    fig, ax = plt.subplots(6, 1, figsize=FIGSIZE_STACKED, sharex=True)

    ax[0].plot(time_s[start:end], gyro_bias[start:end, 0], label=f"Gyro {imu_or_bias} roll")
    ax[1].plot(time_s[start:end], gyro_bias[start:end, 1], label=f"Gyro {imu_or_bias} pitch")
    ax[2].plot(time_s[start:end], gyro_bias[start:end, 2], label=f"Gyro {imu_or_bias} yaw")

    ax[3].plot(time_s[start:end], acc_bias[start:end, 0], label=f"Acc {imu_or_bias} x")
    ax[4].plot(time_s[start:end], acc_bias[start:end, 1], label=f"Acc {imu_or_bias} y")
    ax[5].plot(time_s[start:end], acc_bias[start:end, 2], label=f"Acc {imu_or_bias} z")

    ax[0].set_ylabel("Gyro x [rad/s]")
    ax[1].set_ylabel("Gyro y [rad/s]")
    ax[2].set_ylabel("Gyro z [rad/s]")

    ax[3].set_ylabel("Acc x [m/s²]")
    ax[4].set_ylabel("Acc y [m/s²]")
    ax[5].set_ylabel("Acc z [m/s²]")

    for axis in ax:
        axis.legend()
        axis.grid()

    ax[5].set_xlabel("Elapsed time [s]")

    _set_figure_title(fig, f"{imu_or_bias} Evolution - {comparison_label} - {code}")
    _apply_report_layout(fig=fig)

    filename = f"IMU_{imu_or_bias}_Evolution_{comparison_label}_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, filename))

    if show:
        plt.show()

    plt.close(fig)
    
def load_position_covariance_std_from_msgpack(
    msgpack_path,
    stream_name,
    position_indices=(6, 7, 8),
    fallback_time_stream=None,
):
    """
    Load covariance stream and return timestamps + sqrt(P_xx), sqrt(P_yy), sqrt(P_zz).

    If covariance stream has timestamps, use them.
    If not, use fallback_time_stream, usually pose_body_estimated.
    """

    data = msgpack.load(open(msgpack_path, "rb"))

    if stream_name not in data:
        raise KeyError(
            f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}"
        )

    dataframe = pd.DataFrame(data[stream_name])

    possible_data_columns = [
        "data",
        "covariance.data",
        "state_covariance.data",
        "cov.data",
    ]

    data_column = None
    for col in possible_data_columns:
        if col in dataframe.columns:
            data_column = col
            break

    if data_column is None:
        raise KeyError(
            "Could not find covariance data field. Expected one of "
            f"{possible_data_columns}. Available columns: {list(dataframe.columns)}"
        )

    sigma_xyz = []

    for _, row in dataframe.iterrows():

        rows = int(row["rows"]) if "rows" in dataframe.columns else 15
        cols = int(row["cols"]) if "cols" in dataframe.columns else 15

        P = np.array(row[data_column], dtype=float).reshape(rows, cols)

        sx = np.sqrt(max(P[position_indices[0], position_indices[0]], 0.0))
        sy = np.sqrt(max(P[position_indices[1], position_indices[1]], 0.0))
        sz = np.sqrt(max(P[position_indices[2], position_indices[2]], 0.0))

        sigma_xyz.append([sx, sy, sz])

    sigma_xyz = np.asarray(sigma_xyz)

    # --------------------------------------------------
    # Timestamp handling
    # --------------------------------------------------
    if "time.microseconds" in dataframe.columns:
        timestamps = np.asarray(dataframe["time.microseconds"], dtype=float)

    elif "time" in dataframe.columns:
        timestamps = np.asarray(
            [t["microseconds"] for t in dataframe["time"]],
            dtype=float
        )

    elif fallback_time_stream is not None and fallback_time_stream in data:

        fallback_df = pd.DataFrame(data[fallback_time_stream])

        if "time.microseconds" in fallback_df.columns:
            fallback_timestamps = np.asarray(
                fallback_df["time.microseconds"],
                dtype=float
            )

        elif "time" in fallback_df.columns:
            fallback_timestamps = np.asarray(
                [t["microseconds"] for t in fallback_df["time"]],
                dtype=float
            )

        else:
            raise KeyError(
                f"Fallback time stream '{fallback_time_stream}' has no timestamp field. "
                f"Available columns: {list(fallback_df.columns)}"
            )

        if len(fallback_timestamps) == len(sigma_xyz):
            timestamps = fallback_timestamps

        else:
            # Approximate time axis using pose stream start/end time
            timestamps = np.linspace(
                fallback_timestamps[0],
                fallback_timestamps[-1],
                len(sigma_xyz)
            )

            print(
                "[WARN] Covariance stream has no timestamps and sample count does not "
                "match pose stream. Using interpolated time axis from pose stream."
            )

    else:
        timestamps = np.arange(len(sigma_xyz), dtype=float)

        print(
            "[WARN] Covariance stream has no timestamps and no fallback_time_stream was given. "
            "Using sample index."
        )

    return np.asarray(timestamps), sigma_xyz

def plot_position_covariance_std(
    timestamps,
    sigma_xyz,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
    covariance_label="correction",
):
    """
    Plot sqrt(P_xx), sqrt(P_yy), sqrt(P_zz) over time.
    """

    timestamps = np.asarray(timestamps)
    sigma_xyz = np.asarray(sigma_xyz)

    if timestamps.size == 0:
        raise ValueError("Empty timestamp data. Cannot plot covariance.")

    if sigma_xyz.ndim != 2 or sigma_xyz.shape[1] != 3:
        raise ValueError(f"Expected sigma_xyz shape Nx3, got {sigma_xyz.shape}")

    time_s = elapsed_time_in_seconds(timestamps)

    plt.figure(figsize=FIGSIZE_ONE_THIRD)

    plt.plot(time_s, sigma_xyz[:, 0], label=r"$\sqrt{P_{xx}}$")
    plt.plot(time_s, sigma_xyz[:, 1], label=r"$\sqrt{P_{yy}}$")
    plt.plot(time_s, sigma_xyz[:, 2], label=r"$\sqrt{P_{zz}}$")

    # plt.title(f"Position Covariance Std - {covariance_label} - {comparison_label} - {code}")
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Position std. dev. [m]")
    plt.legend()
    plt.grid()
    _apply_report_layout()

    filename = f"Position_Covariance_Std_{covariance_label}_{comparison_label}_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, filename))

    if show:
        plt.show()

    plt.close()


def calculate_imu_bias_summary(gyro_bias, acc_bias):
    """
    Calculate summary statistics for gyro and accelerometer bias.

    Gyro units: rad/s
    Acc units: m/s^2
    """

    gyro_bias = np.asarray(gyro_bias)
    acc_bias = np.asarray(acc_bias)

    if gyro_bias.size == 0 or acc_bias.size == 0:
        raise ValueError("Empty IMU bias data. Cannot calculate summary.")

    return {
        "samples": int(len(gyro_bias)),

        "gyro_mean_x": float(np.mean(gyro_bias[:, 0])),
        "gyro_mean_y": float(np.mean(gyro_bias[:, 1])),
        "gyro_mean_z": float(np.mean(gyro_bias[:, 2])),

        "gyro_std_x": float(np.std(gyro_bias[:, 0])),
        "gyro_std_y": float(np.std(gyro_bias[:, 1])),
        "gyro_std_z": float(np.std(gyro_bias[:, 2])),

        "gyro_final_x": float(gyro_bias[-1, 0]),
        "gyro_final_y": float(gyro_bias[-1, 1]),
        "gyro_final_z": float(gyro_bias[-1, 2]),

        "acc_mean_x": float(np.mean(acc_bias[:, 0])),
        "acc_mean_y": float(np.mean(acc_bias[:, 1])),
        "acc_mean_z": float(np.mean(acc_bias[:, 2])),

        "acc_std_x": float(np.std(acc_bias[:, 0])),
        "acc_std_y": float(np.std(acc_bias[:, 1])),
        "acc_std_z": float(np.std(acc_bias[:, 2])),

        "acc_final_x": float(acc_bias[-1, 0]),
        "acc_final_y": float(acc_bias[-1, 1]),
        "acc_final_z": float(acc_bias[-1, 2]),
    }

def load_position_xy_covariance_from_msgpack(
    msgpack_path,
    covariance_stream_name,
    fallback_time_stream=None,
    position_indices=(6, 7),
):
    """
    Load 2D position covariance block P_xy from covariance stream.

    InEKF covariance ordering:
        R  : 0, 1, 2
        V  : 3, 4, 5
        P  : 6, 7, 8

    Uses:
        P_xy = P[6:8, 6:8]
    """

    data = msgpack.load(open(msgpack_path, "rb"))

    if covariance_stream_name not in data:
        raise KeyError(
            f"Stream '{covariance_stream_name}' not found in msgpack file: {msgpack_path}"
        )

    covariance_df = pd.DataFrame(data[covariance_stream_name])

    if "data" not in covariance_df.columns:
        raise KeyError(
            f"Could not find covariance data field. Available columns: {list(covariance_df.columns)}"
        )

    covariance_xy = []

    for _, row in covariance_df.iterrows():
        rows = int(row["rows"]) if "rows" in covariance_df.columns else 15
        cols = int(row["cols"]) if "cols" in covariance_df.columns else 15

        P = np.array(row["data"], dtype=float).reshape(rows, cols)

        P_xy = P[np.ix_(position_indices, position_indices)]
        P_xy = 0.5 * (P_xy + P_xy.T)   # symmetrize for numerical safety

        covariance_xy.append(P_xy)

    covariance_xy = np.asarray(covariance_xy)

    # --------------------------------------------------
    # Timestamp handling
    # --------------------------------------------------
    if "time.microseconds" in covariance_df.columns:
        timestamps = np.asarray(covariance_df["time.microseconds"], dtype=float)

    elif "time" in covariance_df.columns:
        timestamps = np.asarray(
            [t["microseconds"] for t in covariance_df["time"]],
            dtype=float
        )

    elif fallback_time_stream is not None and fallback_time_stream in data:

        fallback_df = pd.DataFrame(data[fallback_time_stream])

        if "time.microseconds" in fallback_df.columns:
            fallback_timestamps = np.asarray(
                fallback_df["time.microseconds"],
                dtype=float
            )

        elif "time" in fallback_df.columns:
            fallback_timestamps = np.asarray(
                [t["microseconds"] for t in fallback_df["time"]],
                dtype=float
            )

        else:
            raise KeyError(
                f"Fallback time stream '{fallback_time_stream}' has no timestamp field. "
                f"Available columns: {list(fallback_df.columns)}"
            )

        if len(fallback_timestamps) == len(covariance_xy):
            timestamps = fallback_timestamps
        else:
            timestamps = np.linspace(
                fallback_timestamps[0],
                fallback_timestamps[-1],
                len(covariance_xy)
            )

            print(
                "[WARN] Covariance stream has no timestamps and sample count does not "
                "match pose stream. Using interpolated time axis from pose stream."
            )

    else:
        timestamps = np.arange(len(covariance_xy), dtype=float)

        print(
            "[WARN] Covariance stream has no timestamps and no fallback_time_stream was given. "
            "Using sample index."
        )

    return np.asarray(timestamps), covariance_xy


def _add_covariance_ellipse(
    ax,
    mean_xy,
    covariance_xy,
    n_sigma=1.0,
    alpha=0.18,
    linewidth=1.0,
):
    """
    Add 2D covariance ellipse to an axis.
    """

    eigvals, eigvecs = np.linalg.eigh(covariance_xy)

    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    width = 2.0 * n_sigma * np.sqrt(eigvals[0])
    height = 2.0 * n_sigma * np.sqrt(eigvals[1])

    ellipse = Ellipse(
        xy=mean_xy,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        alpha=alpha,
        linewidth=linewidth,
    )

    ax.add_patch(ellipse)

def plot_trajectory_with_position_covariance_overlay(
    odom_position,
    vicon_position,
    odom_timestamps,
    covariance_timestamps,
    covariance_xy,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
    reference_label="vicon",
    covariance_label="correction",
    n_sigma=1.0,
    max_ellipses=15,
    skip_initial_s=2.0,
):
    """
    Plot 2D trajectory with sparse position covariance ellipses.

    Ellipse center:
        odometry x-y position

    Ellipse shape:
        P_xy = [[P_xx, P_xy],
                [P_yx, P_yy]]

    Notes:
        - Uses correction covariance by default.
        - Draws only a small number of ellipses for readability.
        - Skips first few seconds to avoid large initialization covariance dominating the plot.
    """

    odom_position = np.asarray(odom_position)
    vicon_position = np.asarray(vicon_position)
    odom_timestamps = np.asarray(odom_timestamps, dtype=float)
    covariance_timestamps = np.asarray(covariance_timestamps, dtype=float)
    covariance_xy = np.asarray(covariance_xy)

    if len(odom_position) == 0 or len(covariance_xy) == 0:
        raise ValueError("Empty odometry or covariance data. Cannot plot covariance overlay.")

    # Convert odom timestamps to elapsed seconds
    odom_time_s = np.asarray(elapsed_time_in_seconds(odom_timestamps))

    # --------------------------------------------------
    # Select sparse ellipse locations
    # --------------------------------------------------
    valid_indices = np.where(odom_time_s >= skip_initial_s)[0]

    if len(valid_indices) == 0:
        valid_indices = np.arange(len(odom_position))

    if len(valid_indices) > max_ellipses:
        ellipse_indices = np.linspace(
            valid_indices[0],
            valid_indices[-1],
            max_ellipses
        ).astype(int)
    else:
        ellipse_indices = valid_indices

    # --------------------------------------------------
    # Plot trajectory
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=TRAJ_FIGSIZE)

    ax.plot(
        vicon_position[:, 0],
        vicon_position[:, 1],
        label=reference_label,
        linewidth=TRAJECTORY_LINE_WIDTH,
    )

    ax.plot(
        odom_position[:, 0],
        odom_position[:, 1],
        label=comparison_label,
        linewidth=TRAJECTORY_LINE_WIDTH,
    )

    # Start / end markers, same style as other trajectory plots
    ax.scatter(
        vicon_position[0, 0],
        vicon_position[0, 1],
        color="green",
        marker="o",
        s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
        label=f"{reference_label} start",
        zorder=5,
    )

    ax.scatter(
        vicon_position[-1, 0],
        vicon_position[-1, 1],
        color="red",
        marker="x",
        s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
        label=f"{reference_label} end",
        zorder=5,
    )

    ax.scatter(
        odom_position[0, 0],
        odom_position[0, 1],
        color="green",
        marker="^",
        s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
        label=f"{comparison_label} start",
        zorder=5,
    )

    ax.scatter(
        odom_position[-1, 0],
        odom_position[-1, 1],
        color="red",
        marker="v",
        s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
        label=f"{comparison_label} end",
        zorder=5,
    )

    # --------------------------------------------------
    # Add sparse covariance ellipses
    # --------------------------------------------------
    for i in ellipse_indices:

        odom_ts = odom_timestamps[i]
        nearest_cov_idx = np.argmin(np.abs(covariance_timestamps - odom_ts))

        _add_covariance_ellipse(
            ax,
            mean_xy=odom_position[i, 0:2],
            covariance_xy=covariance_xy[nearest_cov_idx],
            n_sigma=n_sigma,
            alpha=0.35,
            linewidth=1.2,
        )

    ax.plot(
        [],
        [],
        label=f"{n_sigma:.1f}$\\sigma$ position covariance ellipse",
        linewidth=1.2,
    )

    _set_axis_title(
        ax,
        f"2D Trajectory with 1σ Position Covariance Ellipses - {covariance_label} - {code}"
    )
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.set_aspect("equal", adjustable="box")
    _set_trajectory_view(ax, [vicon_position, odom_position])
    ax.grid(True)
    _set_trajectory_legend(ax, fig)
    if TRAJECTORY_LEGEND_LOCATION != 'outside_bottom':
        _apply_report_layout(pad=0.2)

    filename = (
        f"Trajectory_Position_Covariance_Overlay_"
        f"{covariance_label}_{reference_label}_vs_{comparison_label}_{code}.png"
    )

    if save:
        _save_report_figure(
            os.path.join(plots_path, filename),
            fig=fig,
            bbox_inches=TRAJECTORY_SAVE_BBOX_INCHES,
        )
        # print(f"[INFO] Saved covariance overlay plot: {os.path.join(plots_path, filename)}")

    if show:
        plt.show()

    plt.close(fig)
    
def interpolate_vector_stream_to_timestamps(source_timestamps, source_values, target_timestamps):
    """
    Interpolate a Nx3 vector stream to target timestamps.

    Parameters
    ----------
    source_timestamps : np.ndarray
        Source timestamps in microseconds.

    source_values : np.ndarray
        Source Nx3 vector data.

    target_timestamps : np.ndarray
        Target timestamps in microseconds.

    Returns
    -------
    interpolated_values : np.ndarray
        Target-length Nx3 vector data.
    """

    source_timestamps = np.asarray(source_timestamps, dtype=float)
    target_timestamps = np.asarray(target_timestamps, dtype=float)
    source_values = np.asarray(source_values, dtype=float)

    if source_timestamps.size == 0:
        raise ValueError("Empty source timestamp array.")

    if target_timestamps.size == 0:
        raise ValueError("Empty target timestamp array.")

    if source_values.ndim != 2 or source_values.shape[1] != 3:
        raise ValueError(f"Expected source_values shape Nx3, got {source_values.shape}")

    if len(source_timestamps) != len(source_values):
        raise ValueError(
            "source_timestamps and source_values must have the same length. "
            f"Got {len(source_timestamps)} and {len(source_values)}"
        )

    if len(source_timestamps) == 1:
        return np.tile(source_values[0], (len(target_timestamps), 1))

    sort_idx = np.argsort(source_timestamps)
    source_timestamps = source_timestamps[sort_idx]
    source_values = source_values[sort_idx]

    interpolated_values = np.column_stack([
        np.interp(
            target_timestamps,
            source_timestamps,
            source_values[:, axis],
        )
        for axis in range(3)
    ])

    return interpolated_values
   
def plot_body_oriented_imu_bias_overlay_and_sum(
    imu_timestamps,
    gyro_measurements,
    acc_measurements,
    bias_timestamps,
    gyro_bias,
    acc_bias,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
):
    """
    Create three IMU plots:

    1. body_oriented IMU and estimated bias overlaid.
    2. body_oriented IMU - estimated bias.
    3. body_oriented IMU, estimated bias, and (body_oriented IMU - Bias) together.

    Bias is interpolated onto the body_oriented IMU timestamps before plotting.
    """

    imu_timestamps = np.asarray(imu_timestamps, dtype=float)
    gyro_measurements = np.asarray(gyro_measurements, dtype=float)
    acc_measurements = np.asarray(acc_measurements, dtype=float)

    bias_timestamps = np.asarray(bias_timestamps, dtype=float)
    gyro_bias = np.asarray(gyro_bias, dtype=float)
    acc_bias = np.asarray(acc_bias, dtype=float)

    # print(f"[INFO] IMU timestamps shape: {imu_timestamps.shape}")
    # print(f"[INFO] Gyro measurements shape: {gyro_measurements.shape}")
    # print(f"[INFO] Acc measurements shape: {acc_measurements.shape}")
    # print(f"[INFO] Bias timestamps shape: {bias_timestamps.shape}")
    # print(f"[INFO] Gyro bias shape: {gyro_bias.shape}")
    # print(f"[INFO] Acc bias shape: {acc_bias.shape}")

    if gyro_measurements.ndim != 2 or gyro_measurements.shape[1] != 3:
        raise ValueError(f"Expected gyro_measurements shape Nx3, got {gyro_measurements.shape}")

    if acc_measurements.ndim != 2 or acc_measurements.shape[1] != 3:
        raise ValueError(f"Expected acc_measurements shape Nx3, got {acc_measurements.shape}")

    if gyro_bias.ndim != 2 or gyro_bias.shape[1] != 3:
        raise ValueError(f"Expected gyro_bias shape Nx3, got {gyro_bias.shape}")

    if acc_bias.ndim != 2 or acc_bias.shape[1] != 3:
        raise ValueError(f"Expected acc_bias shape Nx3, got {acc_bias.shape}")

    if len(imu_timestamps) != len(gyro_measurements) or len(imu_timestamps) != len(acc_measurements):
        raise ValueError("IMU timestamps, gyro measurements, and acc measurements must have same length.")

    if len(bias_timestamps) != len(gyro_bias) or len(bias_timestamps) != len(acc_bias):
        raise ValueError("Bias timestamps, gyro bias, and acc bias must have same length.")

    # ------------------------------------------------------------------
    # Interpolate bias to body_oriented IMU timestamps
    # ------------------------------------------------------------------
    gyro_bias_interp = interpolate_vector_stream_to_timestamps(
        source_timestamps=bias_timestamps,
        source_values=gyro_bias,
        target_timestamps=imu_timestamps,
    )

    acc_bias_interp = interpolate_vector_stream_to_timestamps(
        source_timestamps=bias_timestamps,
        source_values=acc_bias,
        target_timestamps=imu_timestamps,
    )

    gyro_body_oriented_minus_bias = gyro_measurements - gyro_bias_interp
    acc_body_oriented_minus_bias = acc_measurements - acc_bias_interp

    time_s = elapsed_time_in_seconds(imu_timestamps)
    axis_names = ["x", "y", "z"]

    # print(f"[INFO] Interpolated gyro bias shape: {gyro_bias_interp.shape}")
    # print(f"[INFO] Interpolated acc bias shape: {acc_bias_interp.shape}")
    # print(f"[INFO] Gyro body_oriented minus bias shape: {gyro_body_oriented_minus_bias.shape}")
    # print(f"[INFO] Acc body_oriented minus bias shape: {acc_body_oriented_minus_bias.shape}")

    # ------------------------------------------------------------------
    # Plot 1: body_oriented IMU and bias overlaid
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(6, 1, figsize=FIGSIZE_STACKED, sharex=True)

    for i, axis_name in enumerate(axis_names):
        ax[i].plot(
            time_s,
            gyro_measurements[:, i],
            label=f"body_oriented gyro {axis_name}",
            alpha=0.75,
            linewidth=1.2,
        )
        ax[i].plot(
            time_s,
            gyro_bias_interp[:, i],
            label=f"Gyro bias {axis_name}",
            linestyle="--",
            alpha=0.9,
            linewidth=1.2,
        )
        ax[i].set_ylabel(f"Gyro {axis_name} [rad/s]")
        ax[i].legend(loc="best")
        ax[i].grid()

    for i, axis_name in enumerate(axis_names):
        row = i + 3
        ax[row].plot(
            time_s,
            acc_measurements[:, i],
            label=f"body_oriented acc {axis_name}",
            alpha=0.75,
            linewidth=1.2,
        )
        ax[row].plot(
            time_s,
            acc_bias_interp[:, i],
            label=f"Acc bias {axis_name}",
            linestyle="--",
            alpha=0.9,
            linewidth=1.2,
        )
        ax[row].set_ylabel(f"Acc {axis_name} [m/s²]")
        ax[row].legend(loc="best")
        ax[row].grid()

    ax[5].set_xlabel("Elapsed time [s]")
    _set_figure_title(fig, f"body_oriented IMU and Bias Overlay - {comparison_label} - {code}")
    _apply_report_layout(fig=fig)

    overlay_filename = f"IMU_body_oriented_And_Bias_Overlay_{comparison_label}_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, overlay_filename), fig=fig)

    if show:
        plt.show()

    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: body_oriented IMU - bias
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(6, 1, figsize=FIGSIZE_STACKED, sharex=True)

    for i, axis_name in enumerate(axis_names):
        ax[i].plot(
            time_s,
            gyro_body_oriented_minus_bias[:, i],
            label=f"body_oriented gyro - bias {axis_name}",
            alpha=0.9,
            linewidth=1.2,
        )
        ax[i].set_ylabel(f"Gyro {axis_name} [rad/s]")
        ax[i].legend(loc="best")
        ax[i].grid()

    for i, axis_name in enumerate(axis_names):
        row = i + 3
        ax[row].plot(
            time_s,
            acc_body_oriented_minus_bias[:, i],
            label=f"body_oriented acc - bias {axis_name}",
            alpha=0.9,
            linewidth=1.2,
        )
        ax[row].set_ylabel(f"Acc {axis_name} [m/s²]")
        ax[row].legend(loc="best")
        ax[row].grid()

    ax[5].set_xlabel("Elapsed time [s]")
    _set_figure_title(fig, f"body_oriented IMU Minus Bias - {comparison_label} - {code}")
    _apply_report_layout(fig=fig)

    minus_filename = f"IMU_body_oriented_Minus_Bias_{comparison_label}_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, minus_filename), fig=fig)

    if show:
        plt.show()

    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: body_oriented IMU, bias, and (body_oriented IMU - Bias) together
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(6, 1, figsize=FIGSIZE_STACKED_TALL, sharex=True)

    for i, axis_name in enumerate(axis_names):
        ax[i].plot(
            time_s,
            gyro_measurements[:, i],
            label=f"body_oriented gyro {axis_name}",
            alpha=0.55,
            linewidth=1.0,
        )
        ax[i].plot(
            time_s,
            gyro_bias_interp[:, i],
            label=f"Gyro bias {axis_name}",
            linestyle="--",
            alpha=0.85,
            linewidth=1.1,
        )
        ax[i].plot(
            time_s,
            gyro_body_oriented_minus_bias[:, i],
            label=f"body_oriented gyro - bias {axis_name}",
            linestyle="-.",
            alpha=0.75,
            linewidth=1.4,
        )
        ax[i].set_ylabel(f"Gyro {axis_name} [rad/s]")
        ax[i].legend(loc="best")
        ax[i].grid()

    for i, axis_name in enumerate(axis_names):
        row = i + 3
        ax[row].plot(
            time_s,
            acc_measurements[:, i],
            label=f"body_oriented acc {axis_name}",
            alpha=0.55,
            linewidth=1.0,
        )
        ax[row].plot(
            time_s,
            acc_bias_interp[:, i],
            label=f"Acc bias {axis_name}",
            linestyle="--",
            alpha=0.85,
            linewidth=1.1,
        )
        ax[row].plot(
            time_s,
            acc_body_oriented_minus_bias[:, i],
            label=f"body_oriented acc - bias {axis_name}",
            linestyle="-.",
            alpha=0.75,
            linewidth=1.4,
        )
        ax[row].set_ylabel(f"Acc {axis_name} [m/s²]")
        ax[row].legend(loc="best")
        ax[row].grid()

    ax[5].set_xlabel("Elapsed time [s]")
    _set_figure_title(fig, f"body_oriented IMU, Bias, and body_oriented Minus Bias Combined - {comparison_label} - {code}")
    _apply_report_layout(fig=fig)

    combined_filename = f"IMU_body_oriented_Bias_And_Minus_Combined_{comparison_label}_{code}.png"

    if save:
        _save_report_figure(os.path.join(plots_path, combined_filename), fig=fig)

    if show:
        plt.show()

    plt.close(fig)

    return {
        "imu_timestamps": imu_timestamps,
        "gyro_measurements": gyro_measurements,
        "acc_measurements": acc_measurements,
        "gyro_bias_interpolated": gyro_bias_interp,
        "acc_bias_interpolated": acc_bias_interp,
        "gyro_body_oriented_minus_bias": gyro_body_oriented_minus_bias,
        "acc_body_oriented_minus_bias": acc_body_oriented_minus_bias,
        "overlay_filename": overlay_filename,
        "minus_filename": minus_filename,
        "combined_filename": combined_filename,
    }       
        
def load_md2_distances_from_msgpack(
    msgpack_path,
    stream_name,
    num_wheels=4,
    num_spokes_per_wheel=5,
    fallback_time_stream=None,
):
        """
        Load MD² distances.

        MD² stream format:
            list of samples, each sample is a flat list of length 20.

        Timestamp handling:
            1. If MD² stream has timestamps, use them.
            2. Else use fallback_time_stream timestamps.
            3. If sample counts differ, interpolate over fallback time range.
            4. Else use sample index.
        """

        data = msgpack.load(open(msgpack_path, "rb"))

        if stream_name not in data:
            raise KeyError(
                f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}"
            )

        stream_data = data[stream_name]
        expected_size = num_wheels * num_spokes_per_wheel

        # --------------------------------------------------
        # MD² data handling
        # --------------------------------------------------
        dataframe = pd.DataFrame(stream_data)

        # Your stream is just list-of-list, so this becomes Nx20 dataframe.
        md2_flat = dataframe.to_numpy(dtype=float)

        if md2_flat.ndim != 2:
            raise ValueError(
                f"Expected MD² stream to be 2D, got shape {md2_flat.shape}"
            )

        if md2_flat.shape[1] != expected_size:
            raise ValueError(
                f"Expected {expected_size} MD² values per sample, "
                f"got {md2_flat.shape[1]}"
            )

        md2_values = md2_flat.reshape(
            md2_flat.shape[0],
            num_wheels,
            num_spokes_per_wheel,
        )

        # --------------------------------------------------
        # Timestamp handling, copied from covariance pattern
        # --------------------------------------------------
        if "time.microseconds" in dataframe.columns:
            timestamps = np.asarray(dataframe["time.microseconds"], dtype=float)

        elif "time" in dataframe.columns:
            timestamps = np.asarray(
                [t["microseconds"] for t in dataframe["time"]],
                dtype=float
            )

        elif fallback_time_stream is not None and fallback_time_stream in data:

            fallback_df = pd.DataFrame(data[fallback_time_stream])

            if "time.microseconds" in fallback_df.columns:
                fallback_timestamps = np.asarray(
                    fallback_df["time.microseconds"],
                    dtype=float,
                )

            elif "time" in fallback_df.columns:
                fallback_timestamps = np.asarray(
                    [t["microseconds"] for t in fallback_df["time"]],
                    dtype=float,
                )

            else:
                raise KeyError(
                    f"Fallback time stream '{fallback_time_stream}' has no timestamp field. "
                    f"Available columns: {list(fallback_df.columns)}"
                )

            if len(fallback_timestamps) == len(md2_values):
                timestamps = fallback_timestamps
            else:
                timestamps = np.linspace(
                    fallback_timestamps[0],
                    fallback_timestamps[-1],
                    len(md2_values),
                )

                print(
                    "[WARN] MD² stream has no timestamps and sample count does not "
                    "match fallback stream. Using interpolated time axis from fallback stream."
                )

        else:
            timestamps = np.arange(len(md2_values), dtype=float)

            print(
                "[WARN] MD² stream has no timestamps and no fallback_time_stream was given. "
                "Using sample index."
            )

        return np.asarray(timestamps), md2_values
    
def plot_md2_distances_for_all_wheels(
    timestamps,
    md2_values,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
    chi_square_threshold=None,
    contact_md2_threshold=None,
    wheel_names=None,
    md_viz_cap=100.0
):
    """
    Plot MD² distances for each wheel and spoke.

    md2_values shape:
        (N, num_wheels, num_spokes_per_wheel)
    """

    timestamps = np.asarray(timestamps)
    md2_values = np.asarray(md2_values)

    if md2_values.ndim != 3:
        raise ValueError(
            f"md2_values must have shape (N, num_wheels, num_spokes), "
            f"got {md2_values.shape}"
        )

    num_samples, num_wheels, num_spokes = md2_values.shape

    if wheel_names is None:
        wheel_names = [f"wheel_{i}" for i in range(num_wheels)]

    x_axis = elapsed_time_in_seconds(timestamps)

    for wheel_id in range(num_wheels):
        plt.figure(figsize=figure_size("one_third"))

        for spoke_id in range(num_spokes):
                    
            md2_plot = np.clip(
                md2_values[:, wheel_id, spoke_id],
                None,
                md_viz_cap,
            )
            plt.plot(
                x_axis,
                md2_plot,
                label=f"spoke {spoke_id}",
                linewidth=1.2,
            )

        if chi_square_threshold is not None:
            plt.axhline(
                chi_square_threshold,
                linestyle="--",
                linewidth=1.5,
                label=f"χ² threshold {chi_square_threshold}",
            )

        # plt.title( f"MD² distances for {wheel_names[wheel_id]} - {comparison_label} - {code}"        )
        plt.xlabel("Time [s]")
        plt.ylabel("Mahalanobis distance squared")
        plt.grid(True)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        _apply_report_layout()

        if save:
            filename = make_plot_name(
                f"MD2_Distances_{wheel_names[wheel_id]}",
                None,
                comparison_label,
                code,
            )
            _save_report_figure(os.path.join(plots_path, filename))

        if show:
            plt.show()

        plt.close()
        
def plot_min_md2_distance_per_wheel(
    timestamps,
    md2_values,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
    chi_square_threshold=None,
    wheel_names=None,
    md_viz_cap=100.0
):
    """
    Plot only the minimum MD² distance for each wheel over time.
    This is useful to see which wheels are close to passing the gate.
    """

    timestamps = np.asarray(timestamps)
    md2_values = np.asarray(md2_values)

    num_wheels = md2_values.shape[1]

    if wheel_names is None:
        wheel_names = [f"wheel_{i}" for i in range(num_wheels)]

    x_axis = elapsed_time_in_seconds(timestamps)
    min_md2 = np.min(md2_values, axis=2)

    plt.figure(figsize=figure_size("one_third"))

    for wheel_id in range(num_wheels):
        
        md2_plot = np.clip(
            min_md2[:, wheel_id],
            None,
            md_viz_cap,
        )
        
        plt.plot(
            x_axis,
            md2_plot,
            label=wheel_names[wheel_id],
            linewidth=1.2,
        )

    if chi_square_threshold is not None:
        plt.axhline(
            chi_square_threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"χ² threshold {chi_square_threshold}",
        )

    # plt.title(f"Minimum MD² distance per wheel - {comparison_label} - {code}")
    plt.xlabel("Time [s]")
    plt.ylabel("Minimum Mahalanobis distance squared")
    plt.grid(True)
    plt.legend()
    _apply_report_layout()

    if save:
        filename = make_plot_name(
            "Min_MD2_Distance_Per_Wheel",
            None,
            comparison_label,
            code,
        )
        _save_report_figure(os.path.join(plots_path, filename))

    if show:
        plt.show()

    plt.close()
    

def _extract_timestamps_from_msgpack_stream(data, stream_name):
    """Return timestamp array from a msgpack stream that contains time fields."""

    if stream_name is None or stream_name not in data:
        return None

    dataframe = pd.DataFrame(data[stream_name])

    if "time.microseconds" in dataframe.columns:
        return np.asarray(dataframe["time.microseconds"], dtype=float)

    if "time" in dataframe.columns:
        return np.asarray(
            [t["microseconds"] for t in dataframe["time"]],
            dtype=float,
        )

    return None


def load_slipping_spokes_from_msgpack(
    msgpack_path,
    stream_name,
    fallback_time_stream=None,
):
    """
    Load slipping spoke IDs from a msgpack stream.

    Expected slipping stream format:
        one sample per estimator iteration, containing a vector/list of
        global spoke IDs, for example [] or [0, 7, 12].

    The slipping stream is allowed to have no timestamp. In that case, the
    timestamps are taken from fallback_time_stream, usually
    pose_body_estimated. If the sample counts differ, an interpolated time
    axis over the fallback stream range is used.

    Returns
    -------
    timestamps : np.ndarray
        Timestamp array in microseconds.

    slipping_spokes : list[list[int]]
        One list of global slipping spoke IDs per sample.
    """

    data = msgpack.load(open(msgpack_path, "rb"))

    if stream_name not in data:
        raise KeyError(
            f"Stream '{stream_name}' not found in msgpack file: {msgpack_path}"
        )

    stream_data = data[stream_name]
    slipping_spokes = []

    for sample in stream_data:
        if sample is None:
            slipping_spokes.append([])
            continue

        if isinstance(sample, dict):
            if "data" in sample:
                sample_values = sample["data"]
            elif "spokes" in sample:
                sample_values = sample["spokes"]
            elif "slipping_spokes" in sample:
                sample_values = sample["slipping_spokes"]
            else:
                sample_values = []
        else:
            sample_values = sample

        if sample_values is None:
            slipping_spokes.append([])
        elif isinstance(sample_values, (list, tuple, np.ndarray)):
            slipping_spokes.append([int(value) for value in sample_values])
        else:
            slipping_spokes.append([int(sample_values)])

    # Timestamp handling. Prefer timestamps in the slipping stream if they exist.
    timestamps = _extract_timestamps_from_msgpack_stream(data, stream_name)

    if timestamps is None and fallback_time_stream is not None:
        fallback_timestamps = _extract_timestamps_from_msgpack_stream(
            data,
            fallback_time_stream,
        )

        if fallback_timestamps is not None:
            if len(fallback_timestamps) == len(slipping_spokes):
                timestamps = fallback_timestamps
            else:
                timestamps = np.linspace(
                    fallback_timestamps[0],
                    fallback_timestamps[-1],
                    len(slipping_spokes),
                )
                print(
                    "[WARN] Slipping-spoke stream has no timestamps and sample count does not "
                    "match fallback stream. Using interpolated time axis from fallback stream."
                )

    if timestamps is None:
        timestamps = np.arange(len(slipping_spokes), dtype=float)
        print(
            "[WARN] Slipping-spoke stream has no timestamps and no valid fallback_time_stream "
            "was given. Using sample index."
        )

    return np.asarray(timestamps), slipping_spokes


def plot_slipping_spokes_timeline(
    timestamps,
    slipping_spokes,
    code="unknown",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    comparison_label="contact_velocity_odometry",
    num_wheels=4,
    num_spokes_per_wheel=5,
    wheel_names=None,
):
    """
    Plot slip detections over time for each wheel and spoke.

    Each subplot corresponds to one wheel. The five horizontal levels in each
    subplot correspond to local spoke IDs S0--S4. A marker indicates that the
    corresponding global spoke ID was reported as slipping at that estimator
    iteration.
    """

    timestamps = np.asarray(timestamps)

    if len(timestamps) == 0:
        raise ValueError("Empty timestamp data. Cannot plot slipping spokes.")

    if len(timestamps) != len(slipping_spokes):
        raise ValueError(
            "timestamps and slipping_spokes must have the same length. "
            f"Got {len(timestamps)} and {len(slipping_spokes)}"
        )

    if wheel_names is None:
        wheel_names = [f"wheel {i}" for i in range(num_wheels)]

    x_axis = np.asarray(elapsed_time_in_seconds(timestamps))

    fig, ax = plt.subplots(
        num_wheels,
        1,
        figsize=FIGSIZE_STACKED,
        sharex=True,
        gridspec_kw={"hspace": 0.25},
    )

    if num_wheels == 1:
        ax = [ax]

    slip_x = [[] for _ in range(num_wheels)]
    slip_y = [[] for _ in range(num_wheels)]

    for sample_idx, sample_spokes in enumerate(slipping_spokes):
        for global_spoke_id in sample_spokes:
            global_spoke_id = int(global_spoke_id)
            wheel_id = global_spoke_id // num_spokes_per_wheel
            spoke_id = global_spoke_id % num_spokes_per_wheel

            if wheel_id < 0 or wheel_id >= num_wheels:
                continue

            slip_x[wheel_id].append(x_axis[sample_idx])
            slip_y[wheel_id].append(spoke_id)

    total_slip_marks = 0

    for wheel_id in range(num_wheels):
        for spoke_id in range(num_spokes_per_wheel):
            ax[wheel_id].hlines(
                spoke_id,
                x_axis[0],
                x_axis[-1],
                linewidth=0.6,
                alpha=0.35,
            )

        ax[wheel_id].scatter(
            slip_x[wheel_id],
            slip_y[wheel_id],
            marker="|",
            s=35,
            linewidths=1.0,
            label="slip detected" if wheel_id == 0 else None,
        )

        total_slip_marks += len(slip_x[wheel_id])

        ax[wheel_id].set_ylabel(wheel_names[wheel_id])
        ax[wheel_id].set_yticks(range(num_spokes_per_wheel))
        ax[wheel_id].set_yticklabels([f"S{i}" for i in range(num_spokes_per_wheel)])
        ax[wheel_id].set_ylim(-0.6, num_spokes_per_wheel - 0.4)
        ax[wheel_id].grid(axis="x", alpha=0.35)
        ax[wheel_id].grid(axis="y", alpha=0.20)

    ax[-1].set_xlabel("Elapsed time [s]")

    if total_slip_marks == 0:
        ax[0].text(
            0.5,
            0.5,
            "No slipping spokes reported",
            transform=ax[0].transAxes,
            ha="center",
            va="center",
        )

    handles, labels = ax[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND,
        )

    _set_figure_title(fig, f"Slip Detection Timeline - {comparison_label} - {code}")
    _apply_report_layout(fig=fig)

    filename = make_plot_name(
        "Slip_Detection_Timeline",
        None,
        comparison_label,
        code,
    )

    if save:
        _save_report_figure(os.path.join(plots_path, filename), fig=fig)

    if show:
        plt.show()

    plt.close(fig)

    return {
        "filename": filename,
        "total_slip_marks": int(total_slip_marks),
        "per_wheel_slip_marks": [int(len(values)) for values in slip_x],
    }

def _three_way_orientation_to_degrees(reference_orientation,
                                      comparison_orientation,
                                      baseline_orientation):
    reference_orientation = np.unwrap(np.array(reference_orientation, dtype=float, copy=True), axis=0)
    comparison_orientation = np.unwrap(np.array(comparison_orientation, dtype=float, copy=True), axis=0)
    baseline_orientation = np.unwrap(np.array(baseline_orientation, dtype=float, copy=True), axis=0)

    for orientation in [reference_orientation, comparison_orientation, baseline_orientation]:
        if np.mean(orientation[:, 2]) > np.pi:
            orientation[:, 2] -= 2 * np.pi
        if np.mean(orientation[:, 2]) < -np.pi:
            orientation[:, 2] += 2 * np.pi

    return (
        np.degrees(reference_orientation),
        np.degrees(comparison_orientation),
        np.degrees(baseline_orientation),
    )


def plot_three_way_trajectory_with_heading(
    reference_position,
    comparison_position,
    baseline_position,
    reference_yaw,
    comparison_yaw,
    baseline_yaw,
    code,
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    heading=True,
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    plot_prefix="MULTI_vicon_cvo_baseline",
    start_point=0,
    step=80,
    scale=3,
    width=TRAJECTORY_HEADING_VECTOR_WIDTH,
):
    if heading:
        if reference_yaw is None or comparison_yaw is None or baseline_yaw is None:
            raise ValueError("Yaw arrays must be provided when heading=True")

        if not (
            len(reference_position) == len(comparison_position) == len(baseline_position)
            == len(reference_yaw) == len(comparison_yaw) == len(baseline_yaw)
        ):
            raise ValueError("Position and yaw arrays must have the same length")

    fig, ax = plt.subplots(figsize=TRAJ_FIGSIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(enable=True)

    reference_color = _trajectory_role_color(reference_label, 'reference')
    comparison_color = _trajectory_role_color(comparison_label, 'comparison')
    baseline_color = _trajectory_role_color(baseline_label, 'baseline')

    ax.plot(reference_position[:, 0], reference_position[:, 1],
            label=reference_label, color=reference_color, alpha=TRAJECTORY_REFERENCE_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.plot(comparison_position[:, 0], comparison_position[:, 1],
            label=comparison_label, color=comparison_color, alpha=TRAJECTORY_COMPARISON_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.plot(baseline_position[:, 0], baseline_position[:, 1],
            label=baseline_label, color=baseline_color, alpha=TRAJECTORY_BASELINE_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)

    if heading:
        def plot_heading_vectors(position, yaw, label, color):
            hv = np.array([np.cos(yaw), np.sin(yaw)]).T
            label_added = False

            for idx, (x, y, p, q) in enumerate(zip(position[:, 0], position[:, 1], hv[:, 0], hv[:, 1])):
                if idx % step == 0 and idx > start_point:
                    ax.quiver(
                        x, y, p, q,
                        angles="xy",
                        scale_units="xy",
                        scale=scale,
                        alpha=(TRAJECTORY_REFERENCE_HEADING_ALPHA if color == reference_color else TRAJECTORY_COMPARISON_HEADING_ALPHA if color == comparison_color else TRAJECTORY_BASELINE_HEADING_ALPHA),
                        color=color,
                        width=width,
                        label=f"{label} heading vectors" if (TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND and not label_added) else None,
                    )
                    label_added = True

        plot_heading_vectors(reference_position, reference_yaw, reference_label, reference_color)
        plot_heading_vectors(comparison_position, comparison_yaw, comparison_label, comparison_color)
        plot_heading_vectors(baseline_position, baseline_yaw, baseline_label, baseline_color)

    for marker_idx, position in enumerate([reference_position, comparison_position, baseline_position]):
        ax.scatter(position[0, 0], position[0, 1],
                   color="green", marker="o", s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
                   zorder=6, label="start point" if marker_idx == 0 else None)
        ax.scatter(position[-1, 0], position[-1, 1],
                   color="black", marker="o", s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
                   zorder=6, label="end point" if marker_idx == 0 else None)

    _set_trajectory_view(ax, [reference_position, comparison_position, baseline_position])

    _set_axis_title(ax, f"Trajectory comparison: {reference_label} vs {comparison_label} vs {baseline_label} - {code}")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid()
    _set_trajectory_legend(ax, fig)
    if TRAJECTORY_LEGEND_LOCATION != 'outside_bottom':
        _apply_report_layout(pad=0.2)

    output_path = None
    if save:
        output_path = os.path.join(plots_path, f"{plot_prefix}_trajectory_{code}.png")
        _save_report_figure(output_path, fig=fig, bbox_inches=TRAJECTORY_SAVE_BBOX_INCHES)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def plot_three_way_position_comparison(
    x_axis,
    reference_position,
    comparison_position,
    baseline_position,
    code,
    axis="x",
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    plot_prefix="MULTI_vicon_cvo_baseline",
):
    axis_map = {
        "x": (0, "X"),
        "y": (1, "Y"),
        "z": (2, "Z"),
    }

    if axis not in axis_map:
        raise ValueError("axis must be one of: x, y, z")

    axis_idx, axis_label = axis_map[axis]
    x_axis = elapsed_time_in_seconds(x_axis)

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(x_axis, reference_position[:, axis_idx], label=f"{reference_label} {axis_label}", color="blue")
    plt.plot(x_axis, comparison_position[:, axis_idx], label=f"{comparison_label} {axis_label}", color="red")
    plt.plot(x_axis, baseline_position[:, axis_idx], label=f"{baseline_label} {axis_label}", color="green")

    # plt.title(f"Position Comparison ({axis_label} Axis) - {code}")
    plt.xlabel("Time in seconds")
    plt.ylabel(f"{axis_label} Position (m)")
    plt.legend()
    plt.grid()
    _apply_report_layout()

    output_path = None
    if save:
        output_path = os.path.join(plots_path, f"{plot_prefix}_position_{axis}_{code}.png")
        _save_report_figure(output_path)

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_three_way_orientation_comparison(
    x_axis,
    reference_orientation,
    comparison_orientation,
    baseline_orientation,
    code,
    axis="yaw",
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    plot_prefix="MULTI_vicon_cvo_baseline",
):
    axis_map = {
        "roll": (0, "Roll"),
        "pitch": (1, "Pitch"),
        "yaw": (2, "Yaw"),
    }

    if axis not in axis_map:
        raise ValueError("axis must be one of: roll, pitch, yaw")

    axis_idx, axis_label = axis_map[axis]
    x_axis = elapsed_time_in_seconds(x_axis)

    reference_deg, comparison_deg, baseline_deg = _three_way_orientation_to_degrees(
        reference_orientation,
        comparison_orientation,
        baseline_orientation,
    )

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(x_axis, reference_deg[:, axis_idx], label=f"{reference_label} {axis_label}", color="blue")
    plt.plot(x_axis, comparison_deg[:, axis_idx], label=f"{comparison_label} {axis_label}", color="red")
    plt.plot(x_axis, baseline_deg[:, axis_idx], label=f"{baseline_label} {axis_label}", color="green")

    # plt.title(f"Orientation Comparison ({axis_label}) - {code}")
    plt.xlabel("Time in seconds")
    plt.ylabel(f"{axis_label} Orientation (degrees)")
    plt.legend()
    plt.grid()
    _apply_report_layout()

    output_path = None
    if save:
        output_path = os.path.join(plots_path, f"{plot_prefix}_orientation_{axis}_{code}.png")
        _save_report_figure(output_path)

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_three_way_odometry_comparison(
    reference_timestamps,
    reference_position,
    comparison_position,
    baseline_position,
    reference_orientation_euler,
    comparison_orientation_euler,
    baseline_orientation_euler,
    reference_yaw,
    comparison_yaw,
    baseline_yaw,
    code,
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    plot_prefix="MULTI_vicon_cvo_baseline",
    heading=True,
):
    os.makedirs(plots_path, exist_ok=True)

    saved_paths = []

    path = plot_three_way_trajectory_with_heading(
        reference_position=reference_position,
        comparison_position=comparison_position,
        baseline_position=baseline_position,
        reference_yaw=reference_yaw,
        comparison_yaw=comparison_yaw,
        baseline_yaw=baseline_yaw,
        code=code,
        reference_label=reference_label,
        comparison_label=comparison_label,
        baseline_label=baseline_label,
        heading=heading,
        save=save,
        show=show,
        plots_path=plots_path,
        plot_prefix=plot_prefix,
    )
    if path is not None:
        saved_paths.append(path)

    for axis in ["x", "y", "z"]:
        path = plot_three_way_position_comparison(
            x_axis=reference_timestamps,
            reference_position=reference_position,
            comparison_position=comparison_position,
            baseline_position=baseline_position,
            code=code,
            axis=axis,
            reference_label=reference_label,
            comparison_label=comparison_label,
            baseline_label=baseline_label,
            save=save,
            show=show,
            plots_path=plots_path,
            plot_prefix=plot_prefix,
        )
        if path is not None:
            saved_paths.append(path)

    for axis in ["roll", "pitch", "yaw"]:
        path = plot_three_way_orientation_comparison(
            x_axis=reference_timestamps,
            reference_orientation=reference_orientation_euler,
            comparison_orientation=comparison_orientation_euler,
            baseline_orientation=baseline_orientation_euler,
            code=code,
            axis=axis,
            reference_label=reference_label,
            comparison_label=comparison_label,
            baseline_label=baseline_label,
            save=save,
            show=show,
            plots_path=plots_path,
            plot_prefix=plot_prefix,
        )
        if path is not None:
            saved_paths.append(path)

    return saved_paths




def _unwrap_orientation_deg(orientation):
    orientation = np.unwrap(np.array(orientation, dtype=float, copy=True), axis=0)

    if np.mean(orientation[:, 2]) > np.pi:
        orientation[:, 2] -= 2 * np.pi
    if np.mean(orientation[:, 2]) < -np.pi:
        orientation[:, 2] += 2 * np.pi

    return np.degrees(orientation)


def _three_way_plot_name(plot_kind, reference_label, comparison_label, baseline_label, code, ext="png"):
    return f"{plot_kind}_{reference_label}_vs_{comparison_label}_vs_{baseline_label}_{code}.{ext}"


def _calculate_ape_series(ground_truth, odometry):
    if not ((ground_truth.shape) == (odometry.shape) and ground_truth.shape[1] == 7):
        raise ValueError(
            "APE input arrays must have the same shape and must be Nx7 arrays."
        )

    position_errors = []
    rotation_errors_deg = []
    distances = [0.0]

    for idx in range(len(ground_truth)):
        position_errors.append(fd(ground_truth[idx, :3], odometry[idx, :3], dim=3))
        rotation_errors_deg.append(np.degrees(fga(ground_truth[idx, 3:7], odometry[idx, 3:7])))

        if idx < len(ground_truth) - 1:
            distances.append(
                distances[-1] + fd(ground_truth[idx, :3], ground_truth[idx + 1, :3], dim=3)
            )

    return np.array(distances), np.array(position_errors), np.array(rotation_errors_deg)


def plot_three_way_trajectory_with_heading(
    reference_position,
    comparison_position,
    baseline_position,
    reference_yaw,
    comparison_yaw,
    baseline_yaw,
    code,
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    heading=True,
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    start_point=0,
    step=80,
    scale=3,
    width=TRAJECTORY_HEADING_VECTOR_WIDTH,
):
    if heading:
        if reference_yaw is None or comparison_yaw is None or baseline_yaw is None:
            raise ValueError("Yaw arrays must be provided when heading=True")

        if not (
            len(reference_position) == len(comparison_position) == len(baseline_position)
            == len(reference_yaw) == len(comparison_yaw) == len(baseline_yaw)
        ):
            raise ValueError("Position and yaw arrays must have the same length")

    fig, ax = plt.subplots(figsize=TRAJ_FIGSIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(enable=True)

    reference_color = _trajectory_role_color(reference_label, 'reference')
    comparison_color = _trajectory_role_color(comparison_label, 'comparison')
    baseline_color = _trajectory_role_color(baseline_label, 'baseline')

    ax.plot(reference_position[:, 0], reference_position[:, 1],
            label=reference_label, color=reference_color, alpha=TRAJECTORY_REFERENCE_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.plot(comparison_position[:, 0], comparison_position[:, 1],
            label=comparison_label, color=comparison_color, alpha=TRAJECTORY_COMPARISON_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.plot(baseline_position[:, 0], baseline_position[:, 1],
            label=baseline_label, color=baseline_color, alpha=TRAJECTORY_BASELINE_ALPHA, linewidth=TRAJECTORY_LINE_WIDTH)

    if heading:
        def plot_heading_vectors(position, yaw, label, color):
            heading_vectors = np.array([np.cos(yaw), np.sin(yaw)]).T
            label_added = False

            for idx, (x, y, p, q) in enumerate(
                zip(position[:, 0], position[:, 1], heading_vectors[:, 0], heading_vectors[:, 1])
            ):
                if idx % step == 0 and idx > start_point:
                    ax.quiver(
                        x,
                        y,
                        p,
                        q,
                        angles="xy",
                        scale_units="xy",
                        scale=scale,
                        alpha=(TRAJECTORY_REFERENCE_HEADING_ALPHA if color == reference_color else TRAJECTORY_COMPARISON_HEADING_ALPHA if color == comparison_color else TRAJECTORY_BASELINE_HEADING_ALPHA),
                        color=color,
                        width=width,
                        label=f"{label} heading vectors" if (TRAJECTORY_SHOW_HEADING_VECTOR_LEGEND and not label_added) else None,
                    )
                    label_added = True

        plot_heading_vectors(reference_position, reference_yaw, reference_label, reference_color)
        plot_heading_vectors(comparison_position, comparison_yaw, comparison_label, comparison_color)
        plot_heading_vectors(baseline_position, baseline_yaw, baseline_label, baseline_color)

    for marker_idx, position in enumerate([reference_position, comparison_position, baseline_position]):
        ax.scatter(position[0, 0], position[0, 1],
                   color="green", marker="o", s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
                   zorder=6, label="start point" if marker_idx == 0 else None)
        ax.scatter(position[-1, 0], position[-1, 1],
                   color="black", marker="o", s=TRAJECTORY_ENDPOINT_MARKER_SIZE,
                   zorder=6, label="end point" if marker_idx == 0 else None)

    _set_trajectory_view(ax, [reference_position, comparison_position, baseline_position])

    _set_axis_title(
        ax,
        f"Trajectory comparison: {reference_label} vs {comparison_label} vs {baseline_label} - {code}"
    )
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid()
    _set_trajectory_legend(ax, fig)
    if TRAJECTORY_LEGEND_LOCATION != 'outside_bottom':
        _apply_report_layout(pad=0.2)

    output_path = None
    if save:
        filename = _three_way_plot_name(
            "Trajectory_With_Heading",
            reference_label,
            comparison_label,
            baseline_label,
            code,
        )
        output_path = os.path.join(plots_path, filename)
        _save_report_figure(output_path, fig=fig, bbox_inches=TRAJECTORY_SAVE_BBOX_INCHES)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def plot_three_way_position_comparison(
    x_axis,
    reference_position,
    comparison_position,
    baseline_position,
    code,
    axis="x",
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
):
    axis_map = {
        "x": (0, "X"),
        "y": (1, "Y"),
        "z": (2, "Z"),
    }

    if axis not in axis_map:
        raise ValueError("axis must be one of: x, y, z")

    axis_idx, axis_label = axis_map[axis]
    x_axis = elapsed_time_in_seconds(x_axis)

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(x_axis, reference_position[:, axis_idx], label=f"{reference_label} {axis_label}", color="blue")
    plt.plot(x_axis, comparison_position[:, axis_idx], label=f"{comparison_label} {axis_label}", color="red")
    plt.plot(x_axis, baseline_position[:, axis_idx], label=f"{baseline_label} {axis_label}", color="green")

    # plt.title(f"Position Comparison ({axis_label} Axis) - {code}")
    plt.xlabel("Time in seconds")
    plt.ylabel(f"{axis_label} Position (m)")
    plt.legend()
    plt.grid()
    _apply_report_layout()

    output_path = None
    if save:
        filename = _three_way_plot_name(
            f"{axis}_position_comparison",
            reference_label,
            comparison_label,
            baseline_label,
            code,
        )
        output_path = os.path.join(plots_path, filename)
        _save_report_figure(output_path)

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_three_way_orientation_comparison(
    x_axis,
    reference_orientation,
    comparison_orientation,
    baseline_orientation,
    code,
    axis="yaw",
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
):
    axis_map = {
        "roll": (0, "Roll"),
        "pitch": (1, "Pitch"),
        "yaw": (2, "Yaw"),
    }

    if axis not in axis_map:
        raise ValueError("axis must be one of: roll, pitch, yaw")

    axis_idx, axis_label = axis_map[axis]
    x_axis = elapsed_time_in_seconds(x_axis)

    reference_deg = _unwrap_orientation_deg(reference_orientation)
    comparison_deg = _unwrap_orientation_deg(comparison_orientation)
    baseline_deg = _unwrap_orientation_deg(baseline_orientation)

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(x_axis, reference_deg[:, axis_idx], label=f"{reference_label} {axis_label}", color="blue")
    plt.plot(x_axis, comparison_deg[:, axis_idx], label=f"{comparison_label} {axis_label}", color="red")
    plt.plot(x_axis, baseline_deg[:, axis_idx], label=f"{baseline_label} {axis_label}", color="green")

    # plt.title(f"Orientation Comparison ({axis_label}) - {code}")
    plt.xlabel("Time in seconds")
    plt.ylabel(f"{axis_label} Orientation (degrees)")
    plt.legend()
    plt.grid()
    _apply_report_layout()

    output_path = None
    if save:
        filename = _three_way_plot_name(
            f"{axis}_orientation_comparison",
            reference_label,
            comparison_label,
            baseline_label,
            code,
        )
        output_path = os.path.join(plots_path, filename)
        _save_report_figure(output_path)

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_three_way_odometry_comparison(
    reference_timestamps,
    reference_position,
    comparison_position,
    baseline_position,
    reference_orientation_euler,
    comparison_orientation_euler,
    baseline_orientation_euler,
    reference_yaw,
    comparison_yaw,
    baseline_yaw,
    code,
    reference_label="vicon",
    comparison_label="contact_velocity_odometry",
    baseline_label="baseline_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    heading=True,
):
    os.makedirs(plots_path, exist_ok=True)

    saved_paths = []

    saved_paths.append(
        plot_three_way_trajectory_with_heading(
            reference_position=reference_position,
            comparison_position=comparison_position,
            baseline_position=baseline_position,
            reference_yaw=reference_yaw,
            comparison_yaw=comparison_yaw,
            baseline_yaw=baseline_yaw,
            code=code,
            reference_label=reference_label,
            comparison_label=comparison_label,
            baseline_label=baseline_label,
            heading=heading,
            save=save,
            show=show,
            plots_path=plots_path,
        )
    )

    for axis in ["x", "y", "z"]:
        saved_paths.append(
            plot_three_way_position_comparison(
                x_axis=reference_timestamps,
                reference_position=reference_position,
                comparison_position=comparison_position,
                baseline_position=baseline_position,
                code=code,
                axis=axis,
                reference_label=reference_label,
                comparison_label=comparison_label,
                baseline_label=baseline_label,
                save=save,
                show=show,
                plots_path=plots_path,
            )
        )

    for axis in ["roll", "pitch", "yaw"]:
        saved_paths.append(
            plot_three_way_orientation_comparison(
                x_axis=reference_timestamps,
                reference_orientation=reference_orientation_euler,
                comparison_orientation=comparison_orientation_euler,
                baseline_orientation=baseline_orientation_euler,
                code=code,
                axis=axis,
                reference_label=reference_label,
                comparison_label=comparison_label,
                baseline_label=baseline_label,
                save=save,
                show=show,
                plots_path=plots_path,
            )
        )

    return [path for path in saved_paths if path is not None]

def plot_cvo_vs_baseline_ape_rpe_comparison(
    cvo_reference_timestamps,
    cvo_reference_position,
    cvo_reference_quaternion,
    cvo_position,
    cvo_quaternion,
    baseline_reference_timestamps,
    baseline_reference_position,
    baseline_reference_quaternion,
    baseline_position,
    baseline_quaternion,
    code,
    cvo_label="contact_velocity_odometry",
    baseline_label="skid_odometry",
    save=False,
    show=False,
    plots_path="/opt/workspace/datasets/",
    rpe_window_length=0.25,
):
    """
    Plot CVO vs baseline APE/RPE comparison over elapsed time.

    APE x-axis:
        elapsed time of each synced reference sample.

    RPE x-axis:
        elapsed time of the start sample of each RPE sliding window.
    """

    os.makedirs(plots_path, exist_ok=True)

    cvo_gt = np.hstack((cvo_reference_position, cvo_reference_quaternion))
    cvo_est = np.hstack((cvo_position, cvo_quaternion))

    baseline_gt = np.hstack((baseline_reference_position, baseline_reference_quaternion))
    baseline_est = np.hstack((baseline_position, baseline_quaternion))

    cvo_elapsed_time = np.array(elapsed_time_in_seconds(cvo_reference_timestamps))
    baseline_elapsed_time = np.array(elapsed_time_in_seconds(baseline_reference_timestamps))

    _, cvo_ape_trans, cvo_ape_rot = _calculate_ape_series(cvo_gt, cvo_est)
    _, baseline_ape_trans, baseline_ape_rot = _calculate_ape_series(baseline_gt, baseline_est)

    cvo_rpe_trans, cvo_rpe_rot, _ = calculate_relative_pose_error(
        cvo_gt,
        cvo_est,
        distance_threshold=rpe_window_length,
    )

    baseline_rpe_trans, baseline_rpe_rot, _ = calculate_relative_pose_error(
        baseline_gt,
        baseline_est,
        distance_threshold=rpe_window_length,
    )

    cvo_rpe_elapsed_time = cvo_elapsed_time[:len(cvo_rpe_trans)]
    baseline_rpe_elapsed_time = baseline_elapsed_time[:len(baseline_rpe_trans)]

    saved_paths = []

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(cvo_elapsed_time, cvo_ape_trans, label=cvo_label, color="red")
    plt.plot(baseline_elapsed_time, baseline_ape_trans, label=baseline_label, color="green")
    # plt.title(f"APE Translational Error Comparison - {code}")
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Translation Error [m]")
    plt.grid()
    plt.legend()
    _apply_report_layout()
    if save:
        path = os.path.join(
            plots_path,
            f"APE_translational_comparison_{cvo_label}_vs_{baseline_label}_{code}.png",
        )
        _save_report_figure(path)
        saved_paths.append(path)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(cvo_elapsed_time, cvo_ape_rot, label=cvo_label, color="red")
    plt.plot(baseline_elapsed_time, baseline_ape_rot, label=baseline_label, color="green")
    # plt.title(f"APE Rotational Error Comparison - {code}")
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Rotation Error [deg]")
    plt.grid()
    plt.legend()
    _apply_report_layout()
    if save:
        path = os.path.join(
            plots_path,
            f"APE_rotational_comparison_{cvo_label}_vs_{baseline_label}_{code}.png",
        )
        _save_report_figure(path)
        saved_paths.append(path)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(cvo_rpe_elapsed_time, cvo_rpe_trans, label=cvo_label, color="red")
    plt.plot(baseline_rpe_elapsed_time, baseline_rpe_trans, label=baseline_label, color="green")
    # plt.title(f"RPE Translational Error Comparison - {code}")
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Translation Error [m]")
    plt.grid()
    plt.legend()
    _apply_report_layout()
    if save:
        path = os.path.join(
            plots_path,
            f"RPE_translational_comparison_{cvo_label}_vs_{baseline_label}_{code}.png",
        )
        _save_report_figure(path)
        saved_paths.append(path)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=figure_size("one_third"))
    plt.plot(cvo_rpe_elapsed_time, np.degrees(cvo_rpe_rot), label=cvo_label, color="red")
    plt.plot(baseline_rpe_elapsed_time, np.degrees(baseline_rpe_rot), label=baseline_label, color="green")
    # plt.title(f"RPE Rotational Error Comparison - {code}")
    plt.xlabel("Elapsed time [s]")
    plt.ylabel("Rotation Error [deg]")
    plt.grid()
    plt.legend()
    _apply_report_layout()
    if save:
        path = os.path.join(
            plots_path,
            f"RPE_rotational_comparison_{cvo_label}_vs_{baseline_label}_{code}.png",
        )
        _save_report_figure(path)
        saved_paths.append(path)
    if show:
        plt.show()
    plt.close()

    return saved_paths