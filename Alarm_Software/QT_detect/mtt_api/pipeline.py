import contextlib
import importlib
import os
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import ProcessorConfig
from .models import (
    AudioChunk,
    ClusterAnalysis,
    ClusterResult,
    IdaeResult,
    LinePoint,
    ProcessingResult,
    RecognitionResult,
    RecognitionSegment,
    Trajectory,
)
from .serialization import array_to_payload, flatten_time_frequency


@contextlib.contextmanager
def _project_import_context(project_root: str):
    old_cwd = os.getcwd()
    inserted = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        inserted = True
    os.chdir(project_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(project_root)
            except ValueError:
                pass


class MarineAlgorithmPipeline:
    """Stable API facade over the current QT_detect algorithm scripts.

    Existing scripts are treated as implementation details. This class is the
    boundary that future IPC servers should call from C++.
    """

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.config.validate()

        self._legacy_processor = None
        self._model = None
        self._session_start_time = None
        self._last_relative_end_time = 0.0

    def initialize(self):
        if self._legacy_processor is not None:
            return

        with _project_import_context(self.config.project_root):
            import torch
            from yolo_denoiser import YoloDenoiser

            processor_module = importlib.import_module(self.config.processor_module)
            StreamLineDOAClusterProcessor = processor_module.StreamLineDOAClusterProcessor

            model = YoloDenoiser(out_channel=2)
            state_dict = torch.load(self.config.resolved_model_path(), map_location=self.config.device)
            model.load_state_dict(state_dict)
            model.to(self.config.device)

            processor = StreamLineDOAClusterProcessor(
                fs=1,  # Replaced on the first chunk if needed.
                model=model,
                device=self.config.device,
                process_window=self.config.process_window,
                process_hop=self.config.process_hop,
                raw_buffer_seconds=self.config.raw_buffer_seconds,
                stft_win_seconds=self.config.stft_win_seconds,
                stft_hop_seconds=self.config.stft_hop_seconds,
                doa_delay=self.config.doa_delay,
                doa_win_len=self.config.doa_win_len,
                doa_mode=self.config.doa_mode,
                frequency_resolution=self.config.frequency_resolution,
                f_lower_bound=self.config.f_lower_bound,
                f_higher_bound=self.config.f_higher_bound,
                denoise_thresh=self.config.denoise_thresh,
                line_channel=self.config.line_channel,
                spec_freq_div=self.config.spec_freq_div,
                add_f_lower_bound=self.config.add_f_lower_bound,
                track_match_max_dt=self.config.track_match_max_dt,
                track_match_max_df=self.config.track_match_max_df,
                analysis_window=self.config.analysis_window,
                debug_dir=self.config.debug_dir,
            )

        self._model = model
        self._legacy_processor = processor

    def push_audio(self, chunk: AudioChunk, include_array_data: bool = True) -> ProcessingResult:
        chunk.validate()
        self.initialize()

        if self._session_start_time is None:
            self._session_start_time = float(chunk.start_time)

        # The legacy processor stores fs and time internally. Keep one pipeline
        # instance per buoy/sample-rate pair in the eventual service layer.
        self._legacy_processor.fs = int(chunk.fs)

        raw = self._legacy_processor.push(chunk.pt, chunk.vx, chunk.vy)
        return self._convert_result(chunk, raw, include_array_data=include_array_data)

    @property
    def session_start_time(self) -> Optional[float]:
        return self._session_start_time

    def _convert_result(self, chunk: AudioChunk, raw: Dict, include_array_data: bool) -> ProcessingResult:
        noisy_spec = raw.get("noisy_spec")
        denoise_spec = raw.get("denoise_spec")
        doa_matrix = raw.get("doa_matrix")
        time_azimuth = raw.get("time_azimuth")
        doa_results = raw.get("doa_results") or []
        analysis_result = raw.get("analysis_result")

        spec_for_shape = denoise_spec if denoise_spec is not None else noisy_spec
        rows = 0
        cols = 0
        time_axis: List[float] = []
        freq_axis: List[float] = []
        relative_start = self._last_relative_end_time
        relative_end = self._legacy_processor.current_time

        if spec_for_shape is not None:
            spec_arr = np.asarray(spec_for_shape)
            if spec_arr.ndim != 2:
                raise ValueError("expected 2D spec matrix from legacy processor")
            cols = int(spec_arr.shape[0])
            rows = int(spec_arr.shape[1])
            relative_start = self._infer_output_start_time(cols)
            time_axis = [
                relative_start + i * self.config.stft_hop_seconds
                for i in range(cols)
            ]
            if time_axis:
                relative_end = time_axis[-1]
            freq_axis = self._build_freq_axis(rows)

        line_points = self._convert_line_points(doa_results)
        trajectories = self._group_trajectories(line_points)
        cluster_analysis = self._convert_cluster_analysis(analysis_result)
        clusters = cluster_analysis.clusters if cluster_analysis else self._current_clusters()

        idae = IdaeResult(
            freq_axis=freq_axis,
            time_axis=time_axis,
            rows=rows,
            cols=cols,
            power_db=array_to_payload(
                flatten_time_frequency(denoise_spec if denoise_spec is not None else noisy_spec),
                include_data=include_array_data,
            ),
            doa_matrix=array_to_payload(
                flatten_time_frequency(doa_matrix),
                include_data=include_array_data,
            ),
            noisy_spec=array_to_payload(noisy_spec, include_data=include_array_data),
            denoise_spec=array_to_payload(denoise_spec, include_data=include_array_data),
            time_azimuth=array_to_payload(time_azimuth, include_data=include_array_data),
            line_points=line_points,
            trajectories=trajectories,
        )

        absolute_start = self._session_start_time + relative_start
        absolute_end = self._session_start_time + relative_end
        self._last_relative_end_time = max(self._last_relative_end_time, relative_end)

        recognition = RecognitionResult(
            segments=self._make_recognition_segments(cluster_analysis, absolute_start, absolute_end),
            clusters=clusters,
            cluster_analysis=cluster_analysis,
            target_events=[],
        )

        return ProcessingResult(
            buoy_id=chunk.buoy_id,
            start_time=absolute_start,
            end_time=absolute_end,
            relative_start_time=relative_start,
            relative_end_time=relative_end,
            idae=idae,
            recognition=recognition,
        )

    def _infer_output_start_time(self, cols: int) -> float:
        if cols <= 0:
            return self._last_relative_end_time

        next_process_time = float(getattr(self._legacy_processor, "next_process_time", 0.0))
        last_window_end = next_process_time - self.config.process_hop
        if last_window_end > 0:
            return max(
                0.0,
                last_window_end - self.config.process_hop - self.config.stft_win_seconds / 2.0,
            )

        return self._last_relative_end_time

    def _build_freq_axis(self, rows: int) -> List[float]:
        if rows <= 0:
            return []
        if rows == 1:
            return [float(self.config.f_lower_bound)]
        if self.config.spec_freq_div > 0:
            return [
                float(self.config.f_lower_bound) + float(i) / self.config.spec_freq_div
                for i in range(rows)
            ]
        span = float(self.config.f_higher_bound) - float(self.config.f_lower_bound)
        return [
            float(self.config.f_lower_bound) + span * float(i) / float(rows - 1)
            for i in range(rows)
        ]

    def _convert_line_points(self, doa_results: Iterable[Dict]) -> List[LinePoint]:
        points = []
        for item in doa_results:
            doa = item.get("doa")
            if doa is not None and not np.isfinite(doa):
                doa = None
            points.append(
                LinePoint(
                    track_id=int(item.get("track_id", 0)),
                    time=float(item.get("time", 0.0)),
                    frequency=float(item.get("freq", 0.0)),
                    doa=None if doa is None else float(doa),
                )
            )
        return points

    def _group_trajectories(self, points: List[LinePoint]) -> List[Trajectory]:
        grouped: Dict[int, Trajectory] = {}
        for point in points:
            traj = grouped.setdefault(point.track_id, Trajectory(track_id=point.track_id))
            traj.times.append(point.time)
            traj.frequencies.append(point.frequency)
            traj.doas.append(-1.0 if point.doa is None else point.doa)
        return [grouped[key] for key in sorted(grouped.keys())]

    def _current_clusters(self) -> List[ClusterResult]:
        analyzer = getattr(self._legacy_processor, "cluster_analyzer", None)
        if analyzer is None:
            return []
        try:
            analysis = analyzer.analyze(float(self._legacy_processor.current_time))
        except Exception:
            return []

        clusters = analysis.get("clusters") or []
        result = []
        for index, cluster in enumerate(clusters, start=1):
            result.append(
                ClusterResult(
                    cluster_id=index,
                    track_ids=[int(x) for x in cluster],
                )
            )
        return result

    def _make_recognition_segments(
        self,
        cluster_analysis: Optional[ClusterAnalysis],
        absolute_start: float,
        absolute_end: float,
    ) -> List[RecognitionSegment]:
        """cluster_count > 0 时视为存在目标，生成一个 label=1 的识别段。"""
        if cluster_analysis is None:
            return []
        if cluster_analysis.cluster_count > 0:
            return [RecognitionSegment(
                start_time=absolute_start,
                end_time=absolute_end,
                label=1,
                probability=1.0,
            )]
        return []

    def _convert_cluster_analysis(self, analysis: Optional[Dict]) -> Optional[ClusterAnalysis]:
        if not analysis:
            return None

        clusters = []
        for index, cluster in enumerate(analysis.get("clusters") or [], start=1):
            clusters.append(
                ClusterResult(
                    cluster_id=index,
                    track_ids=[int(x) for x in cluster],
                )
            )

        return ClusterAnalysis(
            track_count=int(analysis.get("track_count", 0)),
            valid_track_count=int(analysis.get("valid_track_count", 0)),
            cluster_count=int(analysis.get("cluster_count", len(clusters))),
            pair_count=int(analysis.get("pair_count", 0)),
            consistent_pair_count=int(analysis.get("consistent_pair_count", 0)),
            clusters=clusters,
        )
