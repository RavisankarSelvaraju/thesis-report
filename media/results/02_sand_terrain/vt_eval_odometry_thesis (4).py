from datetime import datetime
from pathlib import Path
import argparse
import os
import pickle
import re
import shutil
import numpy as np

from vt_eval_odometry_class import OdometryEvaluation
from odometry_evaluation_utils import plot_cvo_vs_baseline_ape_rpe_comparison, plot_three_way_odometry_comparison


SCRIPT_DIR = Path(__file__).resolve().parent

ID2DIR_PKL = SCRIPT_DIR / "dataset-management" / "id2dir.pkl"


def load_pickle_dict(filepath):
    with Path(filepath).open("rb") as f:
        return pickle.load(f)


def load_id2dir_map(pkl_file=ID2DIR_PKL):
    return load_pickle_dict(pkl_file)


def get_dataset_path(dataset_dir, dataset_id):
    match = re.match(r"DATA_(\d+)_\d+", dataset_id)
    if not match:
        raise ValueError(f"Invalid dataset_id format: {dataset_id}")

    xx = int(match.group(1))

    if 18 <= xx <= 28:
        dataset_name = "202605_contact_velocity_odometry_spacehall"
    else:
        dataset_name = "202605_contact_velocity_odometry_mfh"

    return Path(dataset_dir) / dataset_name


def make_output_dir(evaluation_base_dir, folder_prefix, timestamp):
    output_dir = evaluation_base_dir / f"{folder_prefix}_{timestamp}"

    if output_dir.exists():
        print(f"Warning: Output directory already exists and may be overwritten: {output_dir}")
    else:
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)

def copy_log_to_evaluation_dir(dataset_dir_path, log_dir, stream_map, stream_key):
    source_log_path = dataset_dir_path / stream_map[stream_key]["file_name"]
    destination_log_path = log_dir / stream_map[stream_key]["file_name"]

    if not source_log_path.is_file():
        raise FileNotFoundError(f"Required log file not found: {source_log_path}")

    temp_path = destination_log_path.with_name(
        f".{destination_log_path.name}.tmp.{os.getpid()}"
    )

    try:
        with open(source_log_path, "rb") as src, open(temp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())

        shutil.copystat(source_log_path, temp_path)

        if temp_path.stat().st_size != source_log_path.stat().st_size:
            raise RuntimeError(
                f"Copy size mismatch: {source_log_path} -> {temp_path}"
            )

        os.replace(temp_path, destination_log_path)

        print(f"Copied {source_log_path} -> {destination_log_path}")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    
def copy_properties_config(output_dir):
    properties_config_file = Path(
        "/opt/workspace/bundles/coyote3_mcs/config/orogen/contact_velocity_odometry_estimator::Task.yml"
    )

    if properties_config_file.is_file():
        shutil.copy2(properties_config_file, output_dir)
        print(f"Copied properties config to: {output_dir}")
    else:
        print(f"Warning: Properties config file not found: {properties_config_file}")


def create_stream_map():
    return {
        "vicon ground truth": {
            "file_name": "coyote3_vicon_Logger.0.log",
            "stream": "coyote3_vicon.pose_samples",
        },

        "baseline odometry": {
            "file_name": "coyote3_odometry_Logger.0.log",
            "stream": "coyote3_odometry.odometry_samples",
        },

        "contact_odometry": {
            "file_name": "coyote3_contact_aided_pose_estimator_Logger.0.log",
            "stream": "coyote3_contact_aided_pose_estimator.pose_body_estimated",
        },

        "contact velocity odometry": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.pose_body_estimated",
        },

        "combined_odometry": {
            "file_name": "coyote3_combined_odometry_Logger.0.log",
            "stream": "coyote3_combined_odometry.combined_odometry",
        },

        "contact_velocity_odometry_md_contacts": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_md_contact_points",
        },

        "contact_velocity_odometry_lf_contacts": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_lf_contact_points",
        },

        "contact_velocity_odometry_debug_measurement_imu": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_measurement_imu",
        },

        "contact_velocity_odometry_imu_bias": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.imu_bias_estimated",
        },

        "contact_velocity_odometry_covariance_propagation": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_state_covariance_propagation",
        },

        "contact_velocity_odometry_covariance_correction": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_state_covariance_correction",
        },

        "contact_velocity_odometry_md2_distances": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_md2_distances_for_all_wheels",
        },

        "contact_velocity_odometry_slipping_spokes": {
            "file_name": "coyote3_contact_velocity_odometry_estimator_Logger.0.log",
            "stream": "coyote3_contact_velocity_odometry_estimator.debug_slipping_spokes",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run thesis odometry evaluation for CVO, baseline, and multi-comparison plots."
    )

    parser.add_argument(
        "--non-interactive",
        dest="interactive",
        action="store_false",
        help="Kept for bash compatibility. This script currently runs non-interactively.",
    )
    parser.add_argument(
        "--odometry",
        dest="odometry",
        type=str,
        required=True,
        help="Odometry type. Thesis multi evaluation is generated when this is contact velocity odometry.",
    )
    parser.add_argument(
        "--log-dir-path",
        dest="log_dir_path",
        type=str,
        required=True,
        help="Path to the log directory containing replay-generated logs.",
    )
    parser.add_argument(
        "--dataset_id",
        "--dataset-id",
        dest="dataset_id",
        type=str,
        required=True,
        help="Dataset ID, for example DATA_02_01.",
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        dest="dataset_dir",
        type=str,
        default="/202605_contact_velocity_odometry/",
        help="Root directory of the thesis dataset.",
    )

    parser.set_defaults(interactive=False)

    return parser.parse_args()


def plot_multi_comparison_from_evaluators(
    cvo_evaluator,
    baseline_evaluator,
    code,
    output_dir,
):
    """
    Generate Vicon vs CVO vs baseline plots using the already evaluated
    and aligned results from the normal CVO and baseline evaluators.
    """

    cvo_ref_timestamps = [int(t) for t in cvo_evaluator.filtered_reference_timestamps]
    baseline_ref_timestamps = [int(t) for t in baseline_evaluator.filtered_reference_timestamps]

    cvo_index_by_ref_time = {
        ts: idx
        for idx, ts in enumerate(cvo_ref_timestamps)
    }

    baseline_index_by_ref_time = {
        ts: idx
        for idx, ts in enumerate(baseline_ref_timestamps)
    }

    common_reference_timestamps = sorted(
        set(cvo_index_by_ref_time.keys())
        & set(baseline_index_by_ref_time.keys())
    )

    if len(common_reference_timestamps) == 0:
        raise ValueError(
            "No common Vicon reference timestamps found between CVO and baseline evaluation."
        )

    cvo_indices = np.array([
        cvo_index_by_ref_time[ts]
        for ts in common_reference_timestamps
    ])

    baseline_indices = np.array([
        baseline_index_by_ref_time[ts]
        for ts in common_reference_timestamps
    ])

    reference_timestamps = np.array(common_reference_timestamps)

    reference_position = cvo_evaluator.filtered_reference_position[cvo_indices]
    reference_orientation_euler = cvo_evaluator.filtered_reference_orientation_euler[cvo_indices]
    reference_yaw = np.array(cvo_evaluator.filtered_reference_yaw)[cvo_indices]

    cvo_position = cvo_evaluator.filtered_comparison_position[cvo_indices]
    cvo_orientation_euler = cvo_evaluator.filtered_comparison_orientation_euler[cvo_indices]
    cvo_yaw = np.array(cvo_evaluator.filtered_comparison_yaw)[cvo_indices]

    baseline_position = baseline_evaluator.filtered_comparison_position[baseline_indices]
    baseline_orientation_euler = baseline_evaluator.filtered_comparison_orientation_euler[baseline_indices]
    baseline_yaw = np.array(baseline_evaluator.filtered_comparison_yaw)[baseline_indices]

    return plot_three_way_odometry_comparison(
        reference_timestamps=reference_timestamps,
        reference_position=reference_position,
        comparison_position=cvo_position,
        baseline_position=baseline_position,
        reference_orientation_euler=reference_orientation_euler,
        comparison_orientation_euler=cvo_orientation_euler,
        baseline_orientation_euler=baseline_orientation_euler,
        reference_yaw=reference_yaw,
        comparison_yaw=cvo_yaw,
        baseline_yaw=baseline_yaw,
        code=code,
        reference_label="vicon ground truth",
        comparison_label="contact velocity odometry",
        baseline_label="baseline odometry",
        save=True,
        show=False,
        plots_path=output_dir,
        heading=True,
    )

def run_cvo_evaluation(args, stream_map, cvo_output_dir, multi_output_dir):
    reference = "vicon ground truth"
    comparison = "contact velocity odometry"

    evaluator = OdometryEvaluation(
        evaluation_dir_path=args.log_dir_path,
        reference_log_file=stream_map[reference]["file_name"],
        reference_stream=stream_map[reference]["stream"],
        reference_label=reference,

        comparison_log_file=stream_map[comparison]["file_name"],
        comparison_stream=stream_map[comparison]["stream"],
        comparison_label=comparison,

        output_dir=cvo_output_dir,
        match_initial_pose=True,
        code=args.dataset_id,

        enable_contact_metrics=True,
        lf_contact_stream=stream_map["contact_velocity_odometry_lf_contacts"]["stream"],
        md_contact_stream=stream_map["contact_velocity_odometry_md_contacts"]["stream"],
        contact_max_time_diff_us=10000,

        enable_bias_plots=True,
        imu_bias_stream=stream_map["contact_velocity_odometry_imu_bias"]["stream"],

        enable_covariance_plots=True,
        covariance_propagation_stream=stream_map["contact_velocity_odometry_covariance_propagation"]["stream"],
        covariance_correction_stream=stream_map["contact_velocity_odometry_covariance_correction"]["stream"],

        enable_body_oriented_imu_plots=True,
        body_oriented_imu_stream=stream_map["contact_velocity_odometry_debug_measurement_imu"]["stream"],

        enable_md2_distance_plots=True,
        md2_distance_stream=stream_map["contact_velocity_odometry_md2_distances"]["stream"],
        num_wheels=4,
        num_spokes_per_wheel=5,
        chi_square_threshold=1.0,
        contact_md2_threshold=None,
        md_viz_cap=15.0,

        enable_slip_detection_plots=True,
        slipping_spokes_stream=stream_map["contact_velocity_odometry_slipping_spokes"]["stream"],
    )

    evaluator.evaluate()
    return evaluator


def run_baseline_evaluation(args, stream_map, baseline_output_dir):
    reference = "vicon ground truth"
    comparison = "baseline odometry"

    evaluator = OdometryEvaluation(
        evaluation_dir_path=args.log_dir_path,
        reference_log_file=stream_map[reference]["file_name"],
        reference_stream=stream_map[reference]["stream"],
        reference_label=reference,

        comparison_log_file=stream_map[comparison]["file_name"],
        comparison_stream=stream_map[comparison]["stream"],
        comparison_label=comparison,

        output_dir=baseline_output_dir,
        match_initial_pose=True,
        code=args.dataset_id,
    )

    evaluator.evaluate()
    return evaluator



def _latex_value(value, precision=2):
    """Format one numeric value for a LaTeX table."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"

    if not np.isfinite(value):
        return "--"

    return f"{value:.{precision}f}"


def _get_nested_value(dictionary, keys, default=np.nan):
    """Safely read nested values from evaluator.results['summary']."""
    current = dictionary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _format_metric(evaluator, keys, unit="", precision=2):
    value = _get_nested_value(evaluator.results.get("summary", {}), keys)
    value_text = _latex_value(value, precision=precision)

    if value_text == "--" or unit == "":
        return value_text

    return f"{value_text}~{unit}"


def write_latex_metrics_comparison_table(
    baseline_evaluator,
    cvo_evaluator,
    output_dir,
    code,
):
    """
    Write a LaTeX-ready metrics table comparing baseline odometry first
    and contact velocity odometry second.
    """

    # Original with all the metrics
    # rows = [
    #     ("Vicon travel distance", ["travel", "distance_vicon ground truth"], "m", 2),
    #     ("Estimated travel distance", None, "m", 2),
    #     ("Travel distance error", ["travel", "distance_error"], "m", 2),
    #     ("Travel distance error", ["travel", "percentage_error"], r"\%", 2),
    #     ("Final position error", ["travel", "final_error"], "m", 2),
    #     (r"APE translational RMSE", ["position", "rmse"], "m", 2),
    #     (r"APE translational mean", ["position", "mean"], "m", 2),
    #     (r"APE translational max", ["position", "max"], "m", 2),
    #     (r"APE rotational RMSE", ["geodesic_angle", "rmse"], r"$^\circ$", 2),
    #     (r"APE rotational mean", ["geodesic_angle", "mean"], r"$^\circ$", 2),
    #     (r"APE rotational max", ["geodesic_angle", "max"], r"$^\circ$", 2),
    #     (r"Yaw RMSE", ["orientation", "yaw", "rmse"], r"$^\circ$", 2),
    #     (r"X position RMSE", ["per_axis", "x", "rmse"], "m", 2),
    #     (r"Y position RMSE", ["per_axis", "y", "rmse"], "m", 2),
    #     (r"Z position RMSE", ["per_axis", "z", "rmse"], "m", 2),
    #     (r"Roll RMSE", ["orientation", "roll", "rmse"], r"$^\circ$", 2),
    #     (r"Pitch RMSE", ["orientation", "pitch", "rmse"], r"$^\circ$", 2),
    #     (r"RPE translational RMSE", ["rpe_translation", "rmse"], "m", 2),
    #     (r"RPE translational drift RMSE", ["rpe_translation", "rmse_per_meter"], "m/m", 2),
    #     (r"RPE rotational RMSE", ["rpe_rotation", "rmse_deg"], r"$^\circ$", 2),
    #     (r"RPE rotational drift RMSE", ["rpe_rotation", "rmse_deg_per_meter"], r"$^\circ$/m", 2),
    # ]
    
    rows = [
        # ("Vicon travel distance", ["travel", "distance_vicon ground truth"], "m", 2),
        # ("Estimated travel distance", None, "m", 2),
        ("Travel distance error", ["travel", "distance_error"], "m", 2),
        # ("Travel distance error", ["travel", "percentage_error"], r"\%", 2),
        ("Final position error", ["travel", "final_error"], "m", 2),
        # (r"APE translational RMSE", ["position", "rmse"], "m", 2),
        # (r"APE translational mean", ["position", "mean"], "m", 2),
        # (r"APE translational max", ["position", "max"], "m", 2),
        # (r"APE rotational RMSE", ["geodesic_angle", "rmse"], r"$^\circ$", 2),
        # (r"APE rotational mean", ["geodesic_angle", "mean"], r"$^\circ$", 2),
        # (r"APE rotational max", ["geodesic_angle", "max"], r"$^\circ$", 2),
        (r"Yaw RMSE", ["orientation", "yaw", "rmse"], r"$^\circ$", 2),
        # (r"X position RMSE", ["per_axis", "x", "rmse"], "m", 2),
        # (r"Y position RMSE", ["per_axis", "y", "rmse"], "m", 2),
        (r"Z position RMSE", ["per_axis", "z", "rmse"], "m", 2),
        # (r"Roll RMSE", ["orientation", "roll", "rmse"], r"$^\circ$", 2),
        # (r"Pitch RMSE", ["orientation", "pitch", "rmse"], r"$^\circ$", 2),
        (r"RPE translational RMSE", ["rpe_translation", "rmse"], "m", 2),
        (r"RPE translational drift RMSE", ["rpe_translation", "rmse_per_meter"], "m/m", 2),
        (r"RPE rotational RMSE", ["rpe_rotation", "rmse_deg"], r"$^\circ$", 2),
        (r"RPE rotational drift RMSE", ["rpe_rotation", "rmse_deg_per_meter"], r"$^\circ$/m", 2),
    ]

    baseline_summary = baseline_evaluator.results.get("summary", {})
    cvo_summary = cvo_evaluator.results.get("summary", {})

    baseline_distance_key = f"distance_{baseline_evaluator.comparison_label}"
    cvo_distance_key = f"distance_{cvo_evaluator.comparison_label}"

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Odometry evaluation metrics for " + code.replace("_", r"\_") + r".}",
        r"\label{tab:odometry_metrics_" + code.lower().replace("_", "-") + r"}",
        r"\begin{tabular}{lll}",
        r"\hline",
        r"Metric & Baseline odometry & Contact velocity odometry \\",
        r"\hline",
    ]

    for metric_name, keys, unit, precision in rows:
        if metric_name == "Estimated travel distance":
            baseline_text = _latex_value(
                _get_nested_value(baseline_summary, ["travel", baseline_distance_key]),
                precision=precision,
            )
            cvo_text = _latex_value(
                _get_nested_value(cvo_summary, ["travel", cvo_distance_key]),
                precision=precision,
            )
            if baseline_text != "--":
                baseline_text += f"~{unit}"
            if cvo_text != "--":
                cvo_text += f"~{unit}"
        else:
            baseline_text = _format_metric(baseline_evaluator, keys, unit, precision)
            cvo_text = _format_metric(cvo_evaluator, keys, unit, precision)

        latex_lines.append(f"{metric_name} & {baseline_text} & {cvo_text} " + r"\\")
    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    output_path = Path(output_dir) / f"latex_metrics_table_{code}.txt"
    output_path.write_text("\n".join(latex_lines))
    print(f"[INFO] Wrote LaTeX metrics table: {output_path}")

    return output_path

def main():
    args = parse_args()

    if args.odometry != "contact_velocity_odometry":
        raise ValueError(
            "This thesis script is now intended for the CVO thesis evaluation. "
            "Run it with --odometry contact_velocity_odometry."
        )

    log_dir = Path(args.log_dir_path)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Invalid log directory: {log_dir}")

    stream_map = create_stream_map()

    print("Loading dataset map...")
    id2dir = load_id2dir_map()

    dataset_family_path = get_dataset_path(args.dataset_dir, args.dataset_id)
    dataset_dir_path = dataset_family_path / id2dir[args.dataset_id]

    if not dataset_dir_path.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_base_dir = Path(args.dataset_dir) / "evaluation" / args.dataset_id

    cvo_output_dir = make_output_dir(
        evaluation_base_dir,
        "vicon_vs_contact_velocity_odometry",
        timestamp,
    )
    baseline_output_dir = make_output_dir(
        evaluation_base_dir,
        "vicon_vs_baseline_odometry",
        timestamp,
    )
    multi_output_dir = make_output_dir(
        evaluation_base_dir,
        "vicon_vs_contact_velocity_vs_baseline",
        timestamp,
    )

    copy_log_to_evaluation_dir(dataset_dir_path, log_dir, stream_map, "vicon ground truth")
    copy_log_to_evaluation_dir(dataset_dir_path, log_dir, stream_map, "baseline odometry")

    copy_properties_config(cvo_output_dir)
    copy_properties_config(multi_output_dir)

    print("[INFO] Running Vicon vs CVO evaluation and three-way plot generation...")
    cvo_evaluator = run_cvo_evaluation(
        args=args,
        stream_map=stream_map,
        cvo_output_dir=cvo_output_dir,
        multi_output_dir=multi_output_dir,
    )

    print("[INFO] Running Vicon vs baseline evaluation...")
    baseline_evaluator = run_baseline_evaluation(
        args=args,
        stream_map=stream_map,
        baseline_output_dir=baseline_output_dir,
    )

    print("[INFO] Generating Vicon vs CVO vs baseline multi comparison plots...")
    plot_multi_comparison_from_evaluators(
        cvo_evaluator=cvo_evaluator,
        baseline_evaluator=baseline_evaluator,
        code=args.dataset_id,
        output_dir=multi_output_dir,
    )

    print("[INFO] Generating CVO vs baseline APE/RPE comparison plots...")
    plot_cvo_vs_baseline_ape_rpe_comparison(
        cvo_reference_timestamps=cvo_evaluator.filtered_reference_timestamps,
        cvo_reference_position=cvo_evaluator.filtered_reference_position,
        cvo_reference_quaternion=cvo_evaluator.filtered_reference_quaternion,
        cvo_position=cvo_evaluator.filtered_comparison_position,
        cvo_quaternion=cvo_evaluator.filtered_comparison_quaternion,

        baseline_reference_timestamps=baseline_evaluator.filtered_reference_timestamps,
        baseline_reference_position=baseline_evaluator.filtered_reference_position,
        baseline_reference_quaternion=baseline_evaluator.filtered_reference_quaternion,
        baseline_position=baseline_evaluator.filtered_comparison_position,
        baseline_quaternion=baseline_evaluator.filtered_comparison_quaternion,

        code=args.dataset_id,
        cvo_label="contact velocity odometry",
        baseline_label="baseline odometry",
        save=True,
        show=False,
        plots_path=multi_output_dir,
        rpe_window_length=0.25,
    )

    write_latex_metrics_comparison_table(
        baseline_evaluator=baseline_evaluator,
        cvo_evaluator=cvo_evaluator,
        output_dir=multi_output_dir,
        code=args.dataset_id,
    )

    print("[INFO] Thesis evaluation completed.")
    print(f"[INFO] CVO plots: {cvo_output_dir}")
    print(f"[INFO] Baseline plots: {baseline_output_dir}")
    print(f"[INFO] Multi/comparison plots: {multi_output_dir}")


if __name__ == "__main__":
    main()