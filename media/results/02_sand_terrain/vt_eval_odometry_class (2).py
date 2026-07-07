from time import sleep
import os
import json
import numpy as np
from odometry_evaluation_utils import (
    load_debug_measurement_imu_from_msgpack,
    load_position_xy_covariance_from_msgpack,
    plot_body_oriented_imu_bias_overlay_and_sum,
    plot_trajectory_with_position_covariance_overlay,
    relational_msg_to_dict,
    sync_data_streams,
    quaternion_to_euler,
    rotate_trajectory_yaw,
    remove_outliers_from_yaw_data,
    align_initial_pose_se3,
    plot_position_comparison,
    plot_orientation_comparision,
    plot_error_histogram,
    plot_trajectory_with_heading,
    plot_position_error_over_time,
    plot_trajectory_heatmap_time,
    plot_timestamp_synchronization,
    plot_timestamp_comparison,
    calculate_absolute_pose_error,
    calculate_relative_pose_error,
    plot_relative_pose_error_histogram,
    plot_relative_pose_translation_error,
    plot_relative_pose_rotation_error,
    fd,
    fa,
    calculate_travel_distance,
    load_contact_points_from_msgpack,
    calculate_contact_selection_metrics,
    plot_contact_selection_comparison,
    plot_contact_selection_metrics_bar,
    load_imu_bias_from_msgpack,
    plot_imu_and_bias_evolution,
    calculate_imu_bias_summary,
    load_position_covariance_std_from_msgpack,
    plot_position_covariance_std,
    load_md2_distances_from_msgpack,
    plot_md2_distances_for_all_wheels,
    plot_min_md2_distance_per_wheel,
    load_slipping_spokes_from_msgpack,
    plot_slipping_spokes_timeline,
)
from convert_wrapper import convert_pocolog_to_msgpack, convert_msgpack_to_relational


class OdometryEvaluation:
    """Evaluate two log streams and produce the same plots/metrics as vt_eval_odometry.py.

    This class is generic: you can compare any two streams from any two log files,
    including two Vicon logs, two odometry logs, or one Vicon and one odometry log.
    """

    def __init__(
        self,
        evaluation_dir_path,
        reference_log_file,
        reference_stream,
        comparison_log_file,
        comparison_stream,
        code='dataset',
        description=None,
        reference_label='reference',
        comparison_label='comparison',
        save_plots=True,
        show=False,
        enable_heading_in_plot=True,
        align_comparison_with_reference=False,
        comparison_rotation_angle=0.0,
        match_initial_pose=True,
        align_angle=30.0,
        output_dir=None,
        debug=False,
        enable_contact_metrics=False,
        lf_contact_stream=None,
        md_contact_stream=None,
        contact_max_time_diff_us=10000,
        enable_bias_plots=False,
        imu_bias_stream=None,
        enable_covariance_plots=False,
        covariance_propagation_stream=None,
        covariance_correction_stream=None,
        enable_body_oriented_imu_plots=False,
        body_oriented_imu_stream=None,
        enable_md2_distance_plots=False,
        md2_distance_stream=None,
        num_wheels=4,
        num_spokes_per_wheel=5,
        chi_square_threshold=7.815,
        contact_md2_threshold=None,
        md_viz_cap=100.0,
        enable_slip_detection_plots=False,
        slipping_spokes_stream=None,
        
    ):
        self.evaluation_dir_path = os.path.abspath(evaluation_dir_path)
        self.reference_log_file = reference_log_file
        self.reference_stream = reference_stream
        self.comparison_log_file = comparison_log_file
        self.comparison_stream = comparison_stream
        self.code = code
        self.description = description or comparison_label
        self.reference_label = reference_label
        self.comparison_label = comparison_label
        self.save_plots = save_plots
        self.show = show
        self.enable_heading_in_plot = enable_heading_in_plot
        self.align_comparison_with_reference = align_comparison_with_reference
        self.comparison_rotation_angle = comparison_rotation_angle
        self.match_initial_pose = match_initial_pose
        self.align_angle = align_angle
        self.debug = debug
        self.enable_contact_metrics = enable_contact_metrics
        self.lf_contact_stream = lf_contact_stream
        self.md_contact_stream = md_contact_stream
        self.contact_max_time_diff_us = contact_max_time_diff_us
        self.enable_bias_plots = enable_bias_plots
        self.imu_bias_stream = imu_bias_stream
        self.enable_covariance_plots = enable_covariance_plots
        self.covariance_propagation_stream = covariance_propagation_stream
        self.covariance_correction_stream = covariance_correction_stream
        self.enable_body_oriented_imu_plots = enable_body_oriented_imu_plots
        self.body_oriented_imu_stream = body_oriented_imu_stream
        self.enable_md2_distance_plots = enable_md2_distance_plots
        self.md2_distance_stream = md2_distance_stream
        self.num_wheels = num_wheels
        self.num_spokes_per_wheel = num_spokes_per_wheel
        self.chi_square_threshold = chi_square_threshold
        self.contact_md2_threshold = contact_md2_threshold
        self.md_viz_cap = md_viz_cap
        self.enable_slip_detection_plots = enable_slip_detection_plots
        self.slipping_spokes_stream = slipping_spokes_stream

        self.results = {}

        self.output_dir = self._resolve_output_dir(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.reference_log_path = self._resolve_path(reference_log_file)
        self.comparison_log_path = self._resolve_path(comparison_log_file)

    def _resolve_path(self, path_or_name):
        if os.path.isabs(path_or_name):
            return path_or_name
        return os.path.abspath(os.path.join(self.evaluation_dir_path, path_or_name))

    def _resolve_output_dir(self, output_dir):
        if output_dir is None:
            return self.evaluation_dir_path
        if os.path.isabs(output_dir):
            return output_dir
        return os.path.abspath(os.path.join(self.evaluation_dir_path, output_dir))

    def _write_metrics_file(self, text):
        metrics_path = os.path.join(
            self.output_dir,
            f"evaluation_metrics_{self.reference_label}_vs_{self.comparison_label}_{self.code}.txt"
        )
        with open(metrics_path, 'w') as metrics_file:
            metrics_file.write(text)
        if self.debug:      
            print(f"Saved metrics report to: {metrics_path}")
        return metrics_path

    @staticmethod
    def _safe_max(values):
        arr = np.asarray(values)
        return float(np.max(arr)) if arr.size > 0 else float('nan')

    @staticmethod
    def _safe_min(values):
        arr = np.asarray(values)
        return float(np.min(arr)) if arr.size > 0 else float('nan')

    @staticmethod
    def _safe_mean(values):
        arr = np.asarray(values)
        return float(np.mean(arr)) if arr.size > 0 else float('nan')

    @staticmethod
    def _safe_std(values):
        arr = np.asarray(values)
        return float(np.std(arr)) if arr.size > 0 else float('nan')

    @staticmethod
    def _safe_rmse(values):
        arr = np.asarray(values)
        return float(np.sqrt(np.mean(np.square(arr)))) if arr.size > 0 else float('nan')

    def _format_summary_text(self):
        trans_max = self._safe_max(self.results['trans_errors'])
        trans_mean = self._safe_mean(self.results['trans_errors'])
        trans_std = self._safe_std(self.results['trans_errors'])

        rot_max_deg = float(np.degrees(self._safe_max(self.results['rot_error'])))
        rot_mean_deg = float(np.degrees(self._safe_mean(self.results['rot_error'])))
        rot_std_deg = float(np.degrees(self._safe_std(self.results['rot_error'])))
        
        yaw_max = self._safe_max(self.results['yaw_error'])
        yaw_mean = self._safe_mean(self.results['yaw_error'])
        yaw_min = self._safe_min(self.results['yaw_error'])
        yaw_std = self._safe_std(self.results['yaw_error'])
        yaw_rmse = self.results['rmse']['yaw']
        
        lines = [
            '-' * 50,
            'Position error metrics',
            '-' * 50,
            f"Max position error: {self.results['max_position_error']:.2f} m",
            f"Mean position error: {np.mean(self.results['ape_error']['position_errors']):.2f} m",
            f"Min position error: {np.min(self.results['ape_error']['position_errors']):.2f} m",
            f"Std dev of position error: {np.std(self.results['ape_error']['position_errors']):.2f} m",
            f"RMSE of position error: {self.results['ape_error']['position_errors_rmse']:.2f} m",
            '-' * 50,
            'Orientation Geodesic angle error metrics',
            '-' * 50,
            f"Max geodesic angle error: {self.results['max_geodesic_angle_error']:.2f} degrees",
            f"Mean geodesic angle error: {np.mean(self.results['ape_error']['geodesic_angle_errors']):.2f} degrees",
            f"Min geodesic angle error: {np.min(self.results['ape_error']['geodesic_angle_errors']):.2f} degrees",
            f"Std dev of geodesic angle error: {np.std(self.results['ape_error']['geodesic_angle_errors']):.2f} degrees",
            f"RMSE of geodesic angle error: {self.results['ape_error']['geodesic_angle_errors_rmse']:.2f} degrees",
            '-' * 50,
            'Yaw angle error metrics',
            '-' * 50,
            f"Max yaw error: {yaw_max:.2f} degrees",
            f"Mean yaw error: {yaw_mean:.2f} degrees",
            f"Min yaw error: {yaw_min:.2f} degrees",
            f"Std dev of yaw error: {yaw_std:.2f} degrees",
            f"RMSE of yaw error: {yaw_rmse:.2f} degrees",
            '-' * 50,
            'Relative Pose Error (RPE) Translation metrics',
            '-' * 50,
            f"Max translation error: {trans_max:.2f} m",
            f"Mean translation error: {trans_mean:.2f} m",
            f"Std dev of translation error: {trans_std:.2f} m",
            f"RMSE of translation error: {self.results['rmse']['rpe_position']:.2f} m",
            f"Mean translation error per meter: {self.results['rmse']['mean_trans_drift_per_meter']:.2f} m/m",
            f"Std dev of translation error per meter: {(trans_std / 0.25):.2f} m/m",
            f"RMSE of translation error per meter: {self.results['rmse']['trans_drift_per_meter']:.2f} m/m",
            '-' * 50,
            'Relative Pose Error (RPE) Rotation metrics',
            '-' * 50,
            f"Max rotation error: {rot_max_deg:.2f} degrees",
            f"Mean rotation error: {rot_mean_deg:.2f} degrees",
            f"Std dev of rotation error: {rot_std_deg:.2f} degrees",
            f"RMSE of rotation error: {self.results['rmse']['rpe_rotation']:.2f} degrees",
            f"Mean rotation error per meter: {self.results['rmse']['mean_rot_drift']:.2f} degrees/m",
            f"Std dev of rotation error per meter: {(rot_std_deg / 0.25):.2f} degrees/m",
            f"RMSE of rotation error per meter: {self.results['rmse']['rot_drift']:.2f} degrees/m",
            '-' * 50,
        ]
        contact_metrics = self.results.get("contact_selection_metrics", None)

        if contact_metrics is not None:
            overall = contact_metrics["overall"]

            lines.extend([
                '-' * 50,
                'Contact selection metrics: LF proxy vs Mahalanobis',
                '-' * 50,
                f"Contact synced samples: {contact_metrics['synced_samples']}",
                f"Contact precision: {100.0 * overall['precision']:.2f} %",
                f"Contact recall: {100.0 * overall['recall']:.2f} %",
                f"Contact Jaccard: {100.0 * overall['jaccard']:.2f} %",
                f"Contact exact set match: {100.0 * overall['exact_match_rate']:.2f} %",
                f"Mean LF contacts per wheel: {overall['mean_lf_contacts']:.2f}",
                f"Mean MD contacts per wheel: {overall['mean_md_contacts']:.2f}",
                f"Contact selection loss: {overall['contact_selection_loss']:.4f}",
                '-' * 50,
            ])
        imu_bias = self.results.get("imu_bias", None)

        if imu_bias is not None:
            bias_summary = imu_bias["summary"]

            lines.extend([
                '-' * 50,
                'IMU bias estimate metrics',
                '-' * 50,
                f"IMU bias samples: {bias_summary['samples']}",

                f"Mean gyro bias x: {bias_summary['gyro_mean_x']:.8f} rad/s",
                f"Mean gyro bias y: {bias_summary['gyro_mean_y']:.8f} rad/s",
                f"Mean gyro bias z: {bias_summary['gyro_mean_z']:.8f} rad/s",

                f"Std gyro bias x: {bias_summary['gyro_std_x']:.8f} rad/s",
                f"Std gyro bias y: {bias_summary['gyro_std_y']:.8f} rad/s",
                f"Std gyro bias z: {bias_summary['gyro_std_z']:.8f} rad/s",

                f"Final gyro bias x: {bias_summary['gyro_final_x']:.8f} rad/s",
                f"Final gyro bias y: {bias_summary['gyro_final_y']:.8f} rad/s",
                f"Final gyro bias z: {bias_summary['gyro_final_z']:.8f} rad/s",

                f"Mean acc bias x: {bias_summary['acc_mean_x']:.8f} m/s^2",
                f"Mean acc bias y: {bias_summary['acc_mean_y']:.8f} m/s^2",
                f"Mean acc bias z: {bias_summary['acc_mean_z']:.8f} m/s^2",

                f"Std acc bias x: {bias_summary['acc_std_x']:.8f} m/s^2",
                f"Std acc bias y: {bias_summary['acc_std_y']:.8f} m/s^2",
                f"Std acc bias z: {bias_summary['acc_std_z']:.8f} m/s^2",

                f"Final acc bias x: {bias_summary['acc_final_x']:.8f} m/s^2",
                f"Final acc bias y: {bias_summary['acc_final_y']:.8f} m/s^2",
                f"Final acc bias z: {bias_summary['acc_final_z']:.8f} m/s^2",
                '-' * 50,
            ])
        return '\n'.join(lines) + '\n'

    def _format_full_summary_text(self):
        """Return a complete metrics report for file output."""
        lines = [
            'Complete Evaluation Metrics Summary',
            '=' * 50,
            f"Reference label: {self.reference_label}",
            f"Comparison label: {self.comparison_label}",
            f"Dataset code: {self.code}",
            '-' * 50,
            'Human-readable summary',
            '-' * 50,
            self._format_summary_text().rstrip(),
            '-' * 50,
            'Structured summary (JSON)',
            '-' * 50,
            json.dumps(self.results.get('summary', {}), indent=2, sort_keys=True, allow_nan=True),
        ]
        return '\n'.join(lines) + '\n'

    def _ensure_msgpack_path(self, log_path):
        if log_path.endswith('.msgpack'):
            return log_path
        if not log_path.endswith('.log'):
            raise ValueError(f"Unsupported log file extension for {log_path}. Expected .log or .msgpack")
        return log_path[:-4] + '.msgpack'

    def _ensure_relational_path(self, msgpack_path):
        if not msgpack_path.endswith('.msgpack'):
            raise ValueError(f"Expected .msgpack path, got {msgpack_path}")
        return msgpack_path[:-8] + '.relational'

    def _convert_logs(self):
        self.reference_msgpack_path = self._ensure_msgpack_path(self.reference_log_path)
        self.comparison_msgpack_path = self._ensure_msgpack_path(self.comparison_log_path)

        if self.reference_log_path.endswith('.log'):
            convert_pocolog_to_msgpack(self.reference_log_path, self.reference_msgpack_path)

        if self.comparison_log_path.endswith('.log'):
            convert_pocolog_to_msgpack(self.comparison_log_path, self.comparison_msgpack_path)

        msgpack_dirs = {
            os.path.dirname(self.reference_msgpack_path),
            os.path.dirname(self.comparison_msgpack_path),
        }

        for msgpack_dir in msgpack_dirs:
            convert_msgpack_to_relational(msgpack_dir)

    def _load_streams(self):
        self.reference_relational_path = self._ensure_relational_path(self.reference_msgpack_path)
        self.comparison_relational_path = self._ensure_relational_path(self.comparison_msgpack_path)

        self.reference_data = relational_msg_to_dict(
            os.path.dirname(self.reference_relational_path),
            os.path.basename(self.reference_relational_path),
            self.reference_stream,
        )
        self.comparison_data = relational_msg_to_dict(
            os.path.dirname(self.comparison_relational_path),
            os.path.basename(self.comparison_relational_path),
            self.comparison_stream,
        )

    def _compute_and_plot_imu_bias(self):
        """
        Load and plot gyro + accelerometer bias estimate from the comparison estimator log.
        """

        if not self.enable_bias_plots:
            return None

        if self.imu_bias_stream is None:
            if self.debug:
                print("IMU bias plotting requested but imu_bias_stream is not set.")
            return None

        try:
            bias_timestamps, gyro_bias, acc_bias = load_imu_bias_from_msgpack(
                self.comparison_msgpack_path,
                self.imu_bias_stream,
            )

            plot_imu_and_bias_evolution(
                bias_timestamps,
                gyro_bias,
                acc_bias,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                imu_or_bias="bias"
            )

            bias_summary = calculate_imu_bias_summary(
                gyro_bias,
                acc_bias,
            )

            return {
                "timestamps": bias_timestamps,
                "gyro_bias": gyro_bias,
                "acc_bias": acc_bias,
                "summary": bias_summary,
            }

        except Exception as e:
            print(f"IMU bias plot generation failed: {e}")
            return None
    
    def _compute_and_plot_body_oriented_imu(self):
        """
        Load and plot body_oriented IMU measurements from the comparison estimator log.
        """

        if not self.enable_body_oriented_imu_plots:
            return None

        if self.body_oriented_imu_stream is None:
            if self.debug:
                print("body_oriented IMU plotting requested but body_oriented_imu_stream is not set.")
            return None

        try:
            imu_timestamps, gyro_measurements, acc_measurements = load_debug_measurement_imu_from_msgpack(
                self.comparison_msgpack_path,
                self.body_oriented_imu_stream,
            )

            plot_imu_and_bias_evolution(
                imu_timestamps,
                gyro_measurements,
                acc_measurements,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                imu_or_bias="body_oriented_imu"
            )

            return {
                "timestamps": imu_timestamps,
                "gyro_measurements": gyro_measurements,
                "acc_measurements": acc_measurements,
            }

        except Exception as e:
            print(f"body_oriented IMU plot generation failed: {e}")
            return None
    
    
    def _compute_and_plot_position_covariance_std(self):
        """
        Load and plot sqrt(P_xx), sqrt(P_yy), sqrt(P_zz) from covariance streams.
        """

        if not self.enable_covariance_plots:
            return None

        covariance_results = {}

        streams = {
            "propagation": self.covariance_propagation_stream,
            "correction": self.covariance_correction_stream,
        }

        for covariance_label, stream_name in streams.items():

            if stream_name is None:
                if self.debug:
                    print(f"Covariance plotting requested but {covariance_label} stream is not set.")
                continue

            try:
                timestamps, sigma_xyz = load_position_covariance_std_from_msgpack(
                    self.comparison_msgpack_path,
                    stream_name,
                    fallback_time_stream=self.comparison_stream,
                )

                plot_position_covariance_std(
                    timestamps,
                    sigma_xyz,
                    code=self.code,
                    save=self.save_plots,
                    show=self.show,
                    plots_path=self.output_dir,
                    comparison_label=self.comparison_label,
                    covariance_label=covariance_label,
                )

                covariance_results[covariance_label] = {
                    "timestamps": timestamps,
                    "sigma_xyz": sigma_xyz,
                }

            except Exception as e:
                print(f"Position covariance std plot generation failed for {covariance_label}: {e}")

        if len(covariance_results) == 0:
            return None

        return covariance_results
    def _compute_and_plot_trajectory_covariance_overlay(self):
        """
        Plot 2D trajectory with position covariance ellipses overlaid.
        Uses correction covariance by default because it represents the
        covariance after measurement update.
        """

        if not self.enable_covariance_plots:
            return None

        if self.covariance_correction_stream is None:
            if self.debug:
                print("Covariance overlay requested but correction covariance stream is not set.")
            return None

        try:
            covariance_timestamps, covariance_xy = load_position_xy_covariance_from_msgpack(
                self.comparison_msgpack_path,
                self.covariance_correction_stream,
                fallback_time_stream=self.comparison_stream,
            )

            plot_trajectory_with_position_covariance_overlay(
                odom_position=self.filtered_comparison_position,
                vicon_position=self.filtered_reference_position,
                odom_timestamps=self.filtered_comparison_timestamps,
                covariance_timestamps=covariance_timestamps,
                covariance_xy=covariance_xy,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                reference_label=self.reference_label,
                covariance_label="correction",
                n_sigma=1.0,
                max_ellipses=100,
                skip_initial_s=0.0,
            )

            return {
                "timestamps": covariance_timestamps,
                "covariance_xy": covariance_xy,
            }

        except Exception as e:
            print(f"Trajectory covariance overlay plot generation failed: {e}")
            return None
        
    def _compute_and_plot_body_oriented_imu_bias_combined(self):
        """
        Load body_oriented IMU and estimated IMU bias, then create:

        1. body_oriented IMU and bias overlay plot.
        2. body_oriented IMU + bias plot.
        """

        if not self.enable_body_oriented_imu_plots or not self.enable_bias_plots:
            return None

        if self.body_oriented_imu_stream is None:
            if self.debug:
                print("body_oriented IMU + bias plotting requested but body_oriented_imu_stream is not set.")
            return None

        if self.imu_bias_stream is None:
            if self.debug:
                print("body_oriented IMU + bias plotting requested but imu_bias_stream is not set.")
            return None

        try:
            imu_timestamps, gyro_measurements, acc_measurements = load_debug_measurement_imu_from_msgpack(
                self.comparison_msgpack_path,
                self.body_oriented_imu_stream,
            )

            bias_timestamps, gyro_bias, acc_bias = load_imu_bias_from_msgpack(
                self.comparison_msgpack_path,
                self.imu_bias_stream,
            )

            combined_plot_result = plot_body_oriented_imu_bias_overlay_and_sum(
                imu_timestamps=imu_timestamps,
                gyro_measurements=gyro_measurements,
                acc_measurements=acc_measurements,
                bias_timestamps=bias_timestamps,
                gyro_bias=gyro_bias,
                acc_bias=acc_bias,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
            )

            return {
                "imu_timestamps": imu_timestamps,
                "bias_timestamps": bias_timestamps,
                "gyro_measurements": gyro_measurements,
                "acc_measurements": acc_measurements,
                "gyro_bias": gyro_bias,
                "acc_bias": acc_bias,
                "combined_plot_result": combined_plot_result,
            }

        except Exception as e:
            print(f"body_oriented IMU + bias plot generation failed: {e}")
            return None
    
    def _compute_and_plot_md2_distances(self):
        """
        Load and plot MD² distances for all wheels and spokes.
        """

        if not self.enable_md2_distance_plots:
            return None

        if self.md2_distance_stream is None:
            if self.debug:
                print("MD² distance plotting requested but md2_distance_stream is not set.")
            return None

        try:
            timestamps, md2_values = load_md2_distances_from_msgpack(
                self.comparison_msgpack_path,
                self.md2_distance_stream,
                num_wheels=self.num_wheels,
                num_spokes_per_wheel=self.num_spokes_per_wheel,
                fallback_time_stream=self.comparison_stream,
            )

            wheel_names = ["FL", "RL", "FR", "RR"]

            plot_md2_distances_for_all_wheels(
                timestamps,
                md2_values,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                chi_square_threshold=self.chi_square_threshold,
                contact_md2_threshold=self.contact_md2_threshold,
                wheel_names=wheel_names,
                md_viz_cap=self.md_viz_cap,
            )

            plot_min_md2_distance_per_wheel(
                timestamps,
                md2_values,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                chi_square_threshold=self.chi_square_threshold,
                wheel_names=wheel_names,
                md_viz_cap=self.md_viz_cap
            )

            return {
                "timestamps": timestamps,
                "md2_values": md2_values,
            }

        except Exception as e:
            print(f"MD² distance plot generation failed: {e}")
            return None
        
    def _compute_and_plot_slipping_spokes(self):
        """
        Load and plot slipping spoke IDs over time.

        The slipping-spoke stream does not need to contain timestamps; in that
        case, pose_body_estimated timestamps are used as fallback.
        """

        if not self.enable_slip_detection_plots:
            return None

        if self.slipping_spokes_stream is None:
            if self.debug:
                print("Slip detection plotting requested but slipping_spokes_stream is not set.")
            return None

        try:
            timestamps, slipping_spokes = load_slipping_spokes_from_msgpack(
                self.comparison_msgpack_path,
                self.slipping_spokes_stream,
                fallback_time_stream=self.comparison_stream,
            )

            wheel_names = ["FL", "RL", "FR", "RR"]

            plot_result = plot_slipping_spokes_timeline(
                timestamps=timestamps,
                slipping_spokes=slipping_spokes,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                comparison_label=self.comparison_label,
                num_wheels=self.num_wheels,
                num_spokes_per_wheel=self.num_spokes_per_wheel,
                wheel_names=wheel_names,
            )

            return {
                "timestamps": timestamps,
                "slipping_spokes": slipping_spokes,
                "plot": plot_result,
            }

        except Exception as e:
            print(f"Slip detection plot generation failed: {e}")
            return None
        
    def _sync_and_build_arrays(self):
    
        self.synced_comparison_timestamps, self.synced_reference_timestamps = sync_data_streams(
            self.comparison_data,
            self.reference_data,
        )

        self.reference_position = []
        self.comparison_position = []
        self.reference_orientation = []
        self.comparison_orientation = []

        for comp_ts, ref_ts in zip(self.synced_comparison_timestamps, self.synced_reference_timestamps):
            self.comparison_position.append(self.comparison_data[comp_ts]['position'])
            self.reference_position.append(self.reference_data[ref_ts]['position'])
            self.comparison_orientation.append(self.comparison_data[comp_ts]['orientation'])
            self.reference_orientation.append(self.reference_data[ref_ts]['orientation'])

        self.reference_position = np.array(self.reference_position)
        self.comparison_position = np.array(self.comparison_position)
        self.reference_orientation = np.array(self.reference_orientation)
        self.comparison_orientation = np.array(self.comparison_orientation)
    
        
        print("[DEBUG] Reference position first 5 samples:\n", self.reference_position[:5])
        print("[DEBUG] Comparison position first 5 samples:\n", self.comparison_position[:5])
        

        nan_indices = np.unique(np.argwhere(np.isnan(self.comparison_position))[:, 0]) if self.comparison_position.size else np.array([])
        if nan_indices.size > 0:
            if self.debug:
                print(f"Removing NaN values at indices: {nan_indices}")
            self.reference_position = np.delete(self.reference_position, nan_indices, axis=0)
            self.comparison_position = np.delete(self.comparison_position, nan_indices, axis=0)
            self.reference_orientation = np.delete(self.reference_orientation, nan_indices, axis=0)
            self.comparison_orientation = np.delete(self.comparison_orientation, nan_indices, axis=0)
            self.synced_reference_timestamps = np.delete(self.synced_reference_timestamps, nan_indices, axis=0)
            self.synced_comparison_timestamps = np.delete(self.synced_comparison_timestamps, nan_indices, axis=0)

        # if self.align_comparison_with_reference and self.comparison_rotation_angle != 0.0:
        #     self.comparison_position, self.comparison_orientation = rotate_trajectory_yaw(
        #         self.comparison_position,
        #         self.comparison_orientation,
        #         self.comparison_rotation_angle,
        #     )
        
        
        # print(f"Reference position shape: {self.reference_position.shape}")
        # print(f"Comparison position shape: {self.comparison_position.shape}")
        
        # print(f"Reference orientation shape: {self.reference_orientation.shape}")
        # print(f"Comparison orientation shape: {self.comparison_orientation.shape}")

        # # print the first 10 samples of all the data
        # print("First 10 samples of reference position:\n", self.reference_position[:10])
        # print("First 10 samples of comparison position:\n", self.comparison_position[:10])
        # print("First 10 samples of reference orientation:\n", self.reference_orientation[:10])
        # print("First 10 samples of comparison orientation:\n", self.comparison_orientation[:10])

        if self.match_initial_pose:
            self.comparison_position, self.comparison_orientation = align_initial_pose_se3(
                self.comparison_position,
                self.comparison_orientation,
                self.reference_position,
                self.reference_orientation
            )

        self.reference_orientation_euler = np.array([
            quaternion_to_euler(q[0], q[1], q[2], q[3]) for q in self.reference_orientation
        ])
        self.comparison_orientation_euler = np.array([
            quaternion_to_euler(q[0], q[1], q[2], q[3]) for q in self.comparison_orientation
        ])

        self.reference_yaw = self.reference_orientation_euler[:, 2]
        self.comparison_yaw = self.comparison_orientation_euler[:, 2]

        self.filtered_reference_yaw, self.filtered_comparison_yaw, self.filtered_reference_timestamps, _, self.filtered_indices = remove_outliers_from_yaw_data(
            self.reference_yaw,
            self.comparison_yaw,
            self.synced_reference_timestamps,
            debug=self.debug,
            show=self.show,
        )

        if self.filtered_indices.size > 0:
            self.filtered_reference_position = np.delete(self.reference_position, self.filtered_indices, axis=0)
            self.filtered_comparison_position = np.delete(self.comparison_position, self.filtered_indices, axis=0)
            self.filtered_reference_orientation_euler = np.delete(self.reference_orientation_euler, self.filtered_indices, axis=0)
            self.filtered_comparison_orientation_euler = np.delete(self.comparison_orientation_euler, self.filtered_indices, axis=0)
            self.filtered_reference_quaternion = np.delete(self.reference_orientation, self.filtered_indices, axis=0)
            self.filtered_comparison_quaternion = np.delete(self.comparison_orientation, self.filtered_indices, axis=0)
            self.filtered_comparison_timestamps = np.delete(self.synced_comparison_timestamps, self.filtered_indices, axis=0)
        else:
            self.filtered_reference_position = self.reference_position
            self.filtered_comparison_position = self.comparison_position
            self.filtered_reference_orientation_euler = self.reference_orientation_euler
            self.filtered_comparison_orientation_euler = self.comparison_orientation_euler
            self.filtered_reference_quaternion = self.reference_orientation
            self.filtered_comparison_quaternion = self.comparison_orientation
            self.filtered_comparison_timestamps = self.synced_comparison_timestamps

    def _compute_contact_selection_metrics(self):
        """
        Compute LF-vs-MD contact selection metrics.

        LF contact stream is treated as a flat-ground geometric proxy.
        MD contact stream is the Mahalanobis-selected contact estimate.
        """

        if not self.enable_contact_metrics:
            return None

        if self.lf_contact_stream is None or self.md_contact_stream is None:
            if self.debug:
                print("Contact metrics requested but LF/MD contact streams are not set.")
            return None

        try:
            lf_contact_data = load_contact_points_from_msgpack(
                self.comparison_msgpack_path,
                self.lf_contact_stream,
            )

            md_contact_data = load_contact_points_from_msgpack(
                self.comparison_msgpack_path,
                self.md_contact_stream,
            )

            contact_metrics = calculate_contact_selection_metrics(
                lf_contact_data,
                md_contact_data,
                max_time_diff_us=self.contact_max_time_diff_us,
            )

            plot_contact_selection_comparison(
                lf_contact_data,
                md_contact_data,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
                max_time_diff_us=self.contact_max_time_diff_us,
            )

            plot_contact_selection_metrics_bar(
                contact_metrics,
                code=self.code,
                save=self.save_plots,
                show=self.show,
                plots_path=self.output_dir,
            )

            return contact_metrics

        except Exception as e:
            print(f"Contact selection metric calculation failed: {e}")
            return None



    def _plot_and_compute_metrics(self):

        plot_timestamp_synchronization(list(self.reference_data.keys()), 
                                       self.synced_reference_timestamps, 
                                       code=self.code,
                                       stream_name=self.reference_label,
                                       save=self.save_plots,
                                       plots_path=self.output_dir,)

        plot_timestamp_synchronization(list(self.comparison_data.keys()), 
                                       self.synced_comparison_timestamps, 
                                       code=self.code,
                                       stream_name=self.comparison_label,
                                       save=self.save_plots,
                                       plots_path=self.output_dir,)

        plot_timestamp_comparison(
            comparison_timestamps=list(self.comparison_data.keys()),
            reference_timestamps=list(self.reference_data.keys()),
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            save=self.save_plots,
            plots_path=self.output_dir)

        plot_position_comparison(
            self.filtered_reference_timestamps,
            self.filtered_comparison_position,
            self.filtered_reference_position,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='x',
            save=self.save_plots,
            plots_path=self.output_dir,
        )
        plot_position_comparison(
            self.filtered_reference_timestamps,
            self.filtered_comparison_position,
            self.filtered_reference_position,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='y',
            save=self.save_plots,
            plots_path=self.output_dir,
        )
        plot_position_comparison(
            self.filtered_reference_timestamps,
            self.filtered_comparison_position,
            self.filtered_reference_position,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='z',
            save=self.save_plots,
            plots_path=self.output_dir,
        )

        plot_orientation_comparision(
            self.filtered_reference_timestamps,
            self.filtered_comparison_orientation_euler,
            self.filtered_reference_orientation_euler,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='roll',
            save=self.save_plots,
            plots_path=self.output_dir,
        )
        plot_orientation_comparision(
            self.filtered_reference_timestamps,
            self.filtered_comparison_orientation_euler,
            self.filtered_reference_orientation_euler,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='pitch',
            save=self.save_plots,
            plots_path=self.output_dir,
        )
        plot_orientation_comparision(
            self.filtered_reference_timestamps,
            self.filtered_comparison_orientation_euler,
            self.filtered_reference_orientation_euler,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            code=self.code,
            description=self.description,
            axis='yaw',
            save=self.save_plots,
            plots_path=self.output_dir,
        )

        plot_error_histogram(
            self.filtered_reference_orientation_euler,
            self.filtered_comparison_orientation_euler,
            self.code,
            axis='roll',
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            mean_center=False,
            offset=None,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
        )
        plot_error_histogram(
            self.filtered_reference_orientation_euler,
            self.filtered_comparison_orientation_euler,
            self.code,
            axis='pitch',
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            mean_center=False,
            offset=None,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
        )
        plot_error_histogram(
            self.filtered_reference_orientation_euler,
            self.filtered_comparison_orientation_euler,
            self.code,
            axis='yaw',
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            mean_center=False,
            offset=None,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
        )

        plot_trajectory_with_heading(
            self.filtered_comparison_position,
            self.filtered_reference_position,
            self.filtered_comparison_yaw,
            self.filtered_reference_yaw,
            self.code,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            heading=self.enable_heading_in_plot,
            save=self.save_plots,
            show=self.show,
            start_point=0,
            step=100,
            scale=3,
            plots_path=self.output_dir,
        )

        plot_position_error_over_time(
            self.filtered_reference_timestamps,
            self.filtered_comparison_position,
            self.filtered_reference_position,
            self.code,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
        )

        plot_trajectory_heatmap_time(
            self.filtered_comparison_position,
            self.filtered_reference_position,
            self.filtered_comparison_timestamps,
            self.filtered_reference_timestamps,
            self.filtered_comparison_yaw,
            self.filtered_reference_yaw,
            self.code,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            heading=self.enable_heading_in_plot,
            save=self.save_plots,
            show=self.show,
            start_point=600,
            step=80,
            scale=3,
            plots_path=self.output_dir,
        )

        odometry_data = np.hstack((self.filtered_comparison_position, self.filtered_comparison_quaternion))
        ground_truth_data = np.hstack((self.filtered_reference_position, self.filtered_reference_quaternion))

        ape_error = calculate_absolute_pose_error(
            ground_truth_data,
            odometry_data,
            code=self.code,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
        )

        rpe_window_length = 0.25
        trans_errors, rot_error, cumsum_dist = calculate_relative_pose_error(
            ground_truth_data,
            odometry_data,
            distance_threshold=rpe_window_length,
        )

        plot_relative_pose_error_histogram(
            trans_errors,
            rot_error,
            code=self.code,
            save=self.save_plots,
            show=self.show,
            plots_path=self.output_dir,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
        )

        plot_relative_pose_translation_error(
            trans_errors,
            acceptable_max_error=0.5,
            distance_list=cumsum_dist,
            code=self.code,
            plots_path=self.output_dir,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            show=self.show,
            save=self.save_plots,
        )

        plot_relative_pose_rotation_error(
            rot_error,
            acceptable_max_error=np.degrees(0.25),
            distance_list=cumsum_dist,
            code=self.code,
            plots_path=self.output_dir,
            comparison_label=self.comparison_label,
            reference_label=self.reference_label,
            show=self.show,
            save=self.save_plots,
        )

        max_error = np.max(ape_error['position_errors'])
        max_geo_angle_error = np.max(ape_error['geodesic_angle_errors'])

        final_comparison_error = fd(self.filtered_comparison_position[-1], self.filtered_reference_position[-1], dim=3)
        percentage_error = abs(
            calculate_travel_distance(self.filtered_comparison_position) - calculate_travel_distance(self.filtered_reference_position)
        ) / calculate_travel_distance(self.filtered_reference_position) * 100

        x_error = self.filtered_reference_position[:, 0] - self.filtered_comparison_position[:, 0]
        y_error = self.filtered_reference_position[:, 1] - self.filtered_comparison_position[:, 1]
        z_error = self.filtered_reference_position[:, 2] - self.filtered_comparison_position[:, 2]

        roll_error = np.degrees(fa(self.filtered_reference_orientation_euler[:, 0], self.filtered_comparison_orientation_euler[:, 0]))
        pitch_error = np.degrees(fa(self.filtered_reference_orientation_euler[:, 1], self.filtered_comparison_orientation_euler[:, 1]))
        yaw_error = np.degrees(fa(self.filtered_reference_orientation_euler[:, 2], self.filtered_comparison_orientation_euler[:, 2]))

        rmse_x = np.sqrt(np.mean(np.square(x_error)))
        rmse_y = np.sqrt(np.mean(np.square(y_error)))
        rmse_z = np.sqrt(np.mean(np.square(z_error)))

        rmse_roll = np.sqrt(np.mean(np.square(roll_error)))
        rmse_pitch = np.sqrt(np.mean(np.square(pitch_error)))
        rmse_yaw = np.sqrt(np.mean(np.square(yaw_error)))

        rmse_rpe_position = self._safe_rmse(trans_errors)
        rmse_rpe_rotation = float(np.degrees(self._safe_rmse(rot_error)))

        rmse_trans_drift_per_meter = rmse_rpe_position / rpe_window_length
        rmse_rot_drift = rmse_rpe_rotation / rpe_window_length
        mean_trans_drift_per_meter = self._safe_mean(trans_errors) / rpe_window_length
        mean_rot_drift = float(np.degrees(self._safe_mean(rot_error))) / rpe_window_length

        # detailed per-axis and orientation error arrays
        self.results = {
            'ape_error': ape_error,
            'trans_errors': trans_errors,
            'rot_error': rot_error,
            'cumsum_dist': cumsum_dist,
            'max_position_error': max_error,
            'max_geodesic_angle_error': max_geo_angle_error,
            'percentage_error': percentage_error,

            'x_error': x_error,
            'y_error': y_error,
            'z_error': z_error,

            'roll_error': roll_error,
            'pitch_error': pitch_error,
            'yaw_error': yaw_error,

            'rmse': {
                'x': rmse_x,
                'y': rmse_y,
                'z': rmse_z,
                'roll': rmse_roll,
                'pitch': rmse_pitch,
                'yaw': rmse_yaw,
                'rpe_position': rmse_rpe_position,
                'rpe_rotation': rmse_rpe_rotation,
                'trans_drift_per_meter': rmse_trans_drift_per_meter,
                'rot_drift': rmse_rot_drift,
                'mean_trans_drift_per_meter': mean_trans_drift_per_meter,
                'mean_rot_drift': mean_rot_drift,
            },

            # top-level convenience summaries
            'mean_trans_drift_per_meter': mean_trans_drift_per_meter,
            'mean_rot_drift': mean_rot_drift,

            f'final_{self.comparison_label}_error': final_comparison_error,
            f'filtered_{self.reference_label}_timestamps': self.filtered_reference_timestamps,
            f'filtered_{self.comparison_label}_timestamps': self.filtered_comparison_timestamps,
        }

        # human-friendly summary statistics matching the original vt_eval_odometry script prints
        self.results['summary'] = {
            'position': {
                'max': max_error,
                'mean': float(np.mean(ape_error['position_errors'])),
                'min': float(np.min(ape_error['position_errors'])),
                'std': float(np.std(ape_error['position_errors'])),
                'rmse': float(ape_error['position_errors_rmse']),
            },
            'geodesic_angle': {
                'max': max_geo_angle_error,
                'mean': float(np.mean(ape_error['geodesic_angle_errors'])),
                'min': float(np.min(ape_error['geodesic_angle_errors'])),
                'std': float(np.std(ape_error['geodesic_angle_errors'])),
                'rmse': float(ape_error['geodesic_angle_errors_rmse']),
            },
            'rpe_translation': {
                'max': self._safe_max(trans_errors),
                'mean': self._safe_mean(trans_errors),
                'std': self._safe_std(trans_errors),
                'rmse': float(rmse_rpe_position),
                'mean_per_meter': float(mean_trans_drift_per_meter),
                'std_per_meter': self._safe_std(trans_errors) / rpe_window_length,
                'rmse_per_meter': float(rmse_trans_drift_per_meter),
            },
            'rpe_rotation': {
                'max_deg': float(np.degrees(self._safe_max(rot_error))),
                'mean_deg': float(np.degrees(self._safe_mean(rot_error))),
                'std_deg': float(np.degrees(self._safe_std(rot_error))),
                'rmse_deg': float(rmse_rpe_rotation),
                'mean_deg_per_meter': float(mean_rot_drift),
                'std_deg_per_meter': float(np.degrees(self._safe_std(rot_error)) / rpe_window_length),
                'rmse_deg_per_meter': float(rmse_rot_drift),
            },
            'per_axis': {
                'x': {'max': float(np.max(x_error)), 'mean': float(np.mean(x_error)), 'min': float(np.min(x_error)), 'std': float(np.std(x_error)), 'rmse': float(rmse_x)},
                'y': {'max': float(np.max(y_error)), 'mean': float(np.mean(y_error)), 'min': float(np.min(y_error)), 'std': float(np.std(y_error)), 'rmse': float(rmse_y)},
                'z': {'max': float(np.max(z_error)), 'mean': float(np.mean(z_error)), 'min': float(np.min(z_error)), 'std': float(np.std(z_error)), 'rmse': float(rmse_z)},
            },
            'orientation': {
                'roll': {'max': float(np.max(roll_error)), 'mean': float(np.mean(roll_error)), 'min': float(np.min(roll_error)), 'std': float(np.std(roll_error)), 'rmse': float(rmse_roll)},
                'pitch': {'max': float(np.max(pitch_error)), 'mean': float(np.mean(pitch_error)), 'min': float(np.min(pitch_error)), 'std': float(np.std(pitch_error)), 'rmse': float(rmse_pitch)},
                'yaw': {'max': float(np.max(yaw_error)), 'mean': float(np.mean(yaw_error)), 'min': float(np.min(yaw_error)), 'std': float(np.std(yaw_error)), 'rmse': float(rmse_yaw)},
            },
            'travel': {
                f'distance_{self.comparison_label}': float(calculate_travel_distance(self.filtered_comparison_position)),
                f'distance_{self.reference_label}': float(calculate_travel_distance(self.filtered_reference_position)),
                'distance_error': float(calculate_travel_distance(self.filtered_comparison_position) - calculate_travel_distance(self.filtered_reference_position)),
                'final_error': float(final_comparison_error),
                'percentage_error': float(percentage_error),
            },
        }
        contact_metrics = self._compute_contact_selection_metrics()

        if contact_metrics is not None:
            self.results["contact_selection_metrics"] = contact_metrics
            self.results["summary"]["contact_selection"] = contact_metrics["overall"]
            
        imu_bias_result = self._compute_and_plot_imu_bias()

        if imu_bias_result is not None:
            self.results["imu_bias"] = imu_bias_result
            self.results["summary"]["imu_bias"] = imu_bias_result["summary"]
            
        body_oriented_imu_result = self._compute_and_plot_body_oriented_imu()
        if body_oriented_imu_result is not None:
            self.results["body_oriented_imu"] = body_oriented_imu_result
            
        body_oriented_imu_bias_combined_result = self._compute_and_plot_body_oriented_imu_bias_combined()
        if body_oriented_imu_bias_combined_result is not None:
            self.results["body_oriented_imu_bias_combined"] = body_oriented_imu_bias_combined_result
            
        covariance_result = self._compute_and_plot_position_covariance_std()
        if covariance_result is not None:
            self.results["position_covariance_std"] = covariance_result
            
        covariance_overlay_result = self._compute_and_plot_trajectory_covariance_overlay()

        if covariance_overlay_result is not None:
            self.results["trajectory_position_covariance_overlay"] = covariance_overlay_result
            
        self.results["md2_distances"] = self._compute_and_plot_md2_distances()

        slipping_spokes_result = self._compute_and_plot_slipping_spokes()
        if slipping_spokes_result is not None:
            self.results["slipping_spokes"] = slipping_spokes_result

        self._print_summary()

    def _print_summary(self):
        summary_text = self._format_summary_text()
        print(summary_text, end='')
        self._write_metrics_file(self._format_full_summary_text())

    def evaluate(self):
        """Run the full evaluation pipeline and return a result dictionary."""
        self._convert_logs()
        self._load_streams()
        self._sync_and_build_arrays()
        self._plot_and_compute_metrics()

        return self.results
