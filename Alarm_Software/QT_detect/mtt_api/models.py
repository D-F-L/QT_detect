from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ArrayPayload:
    """Protocol-neutral ndarray representation for future IPC layers."""

    shape: Tuple[int, ...]
    dtype: str
    data: Optional[str] = None
    order: str = "C"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "data": self.data,
            "order": self.order,
        }


@dataclass
class AudioChunk:
    buoy_id: str
    fs: int
    start_time: float
    pt: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    @classmethod
    def from_arrays(cls, buoy_id: str, fs: int, start_time: float,
                    pt: Sequence[float], vx: Sequence[float], vy: Sequence[float]):
        return cls(
            buoy_id=buoy_id,
            fs=int(fs),
            start_time=float(start_time),
            pt=np.asarray(pt, dtype=np.float32).reshape(-1),
            vx=np.asarray(vx, dtype=np.float32).reshape(-1),
            vy=np.asarray(vy, dtype=np.float32).reshape(-1),
        )

    def validate(self):
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if not (len(self.pt) == len(self.vx) == len(self.vy)):
            raise ValueError("pt/vx/vy chunk length mismatch")

    @property
    def duration_seconds(self) -> float:
        if self.fs <= 0:
            return 0.0
        return float(len(self.pt)) / float(self.fs)


@dataclass
class LinePoint:
    track_id: int
    time: float
    frequency: float
    doa: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "time": self.time,
            "frequency": self.frequency,
            "doa": self.doa,
        }


@dataclass
class Trajectory:
    track_id: int
    times: List[float] = field(default_factory=list)
    frequencies: List[float] = field(default_factory=list)
    doas: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "times": self.times,
            "frequencies": self.frequencies,
            "doas": self.doas,
        }


@dataclass
class ClusterResult:
    cluster_id: int
    track_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "track_ids": self.track_ids,
        }


@dataclass
class ClusterAnalysis:
    track_count: int = 0
    valid_track_count: int = 0
    cluster_count: int = 0
    pair_count: int = 0
    consistent_pair_count: int = 0
    clusters: List[ClusterResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_count": self.track_count,
            "valid_track_count": self.valid_track_count,
            "cluster_count": self.cluster_count,
            "pair_count": self.pair_count,
            "consistent_pair_count": self.consistent_pair_count,
            "clusters": [c.to_dict() for c in self.clusters],
        }


@dataclass
class TargetEvent:
    event_type: str
    time: float
    target_id: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "time": self.time,
            "target_id": self.target_id,
            "confidence": self.confidence,
        }


@dataclass
class RecognitionSegment:
    start_time: float
    end_time: float
    label: int
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "probability": self.probability,
        }


@dataclass
class IdaeResult:
    freq_axis: List[float] = field(default_factory=list)
    time_axis: List[float] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    power_db: Optional[ArrayPayload] = None
    doa_matrix: Optional[ArrayPayload] = None
    noisy_spec: Optional[ArrayPayload] = None
    denoise_spec: Optional[ArrayPayload] = None
    time_azimuth: Optional[ArrayPayload] = None
    line_points: List[LinePoint] = field(default_factory=list)
    trajectories: List[Trajectory] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freq_axis": self.freq_axis,
            "time_axis": self.time_axis,
            "rows": self.rows,
            "cols": self.cols,
            "power_db": self.power_db.to_dict() if self.power_db else None,
            "doa_matrix": self.doa_matrix.to_dict() if self.doa_matrix else None,
            "noisy_spec": self.noisy_spec.to_dict() if self.noisy_spec else None,
            "denoise_spec": self.denoise_spec.to_dict() if self.denoise_spec else None,
            "time_azimuth": self.time_azimuth.to_dict() if self.time_azimuth else None,
            "line_points": [p.to_dict() for p in self.line_points],
            "trajectories": [t.to_dict() for t in self.trajectories],
        }


@dataclass
class RecognitionResult:
    segments: List[RecognitionSegment] = field(default_factory=list)
    clusters: List[ClusterResult] = field(default_factory=list)
    cluster_analysis: Optional[ClusterAnalysis] = None
    target_events: List[TargetEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "clusters": [c.to_dict() for c in self.clusters],
            "cluster_analysis": self.cluster_analysis.to_dict() if self.cluster_analysis else None,
            "target_events": [e.to_dict() for e in self.target_events],
        }


@dataclass
class ProcessingResult:
    buoy_id: str
    start_time: float
    end_time: float
    relative_start_time: float
    relative_end_time: float
    idae: IdaeResult
    recognition: RecognitionResult = field(default_factory=RecognitionResult)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buoy_id": self.buoy_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "relative_start_time": self.relative_start_time,
            "relative_end_time": self.relative_end_time,
            "idae": self.idae.to_dict(),
            "recognition": self.recognition.to_dict(),
        }
