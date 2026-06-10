import os
import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON/YAML service config file."""
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML config files") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")

    if not isinstance(data, dict):
        raise ValueError("config file root must be an object")
    return data


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _coerce_like(value: Any, current: Any) -> Any:
    if current is None:
        return value
    if isinstance(current, bool):
        return _as_bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


@dataclass
class ProcessorConfig:
    """Configuration for the long-lived Python algorithm pipeline."""

    project_root: str
    model_path: str
    processor_module: str = "DOA_stream_realtime_cluster_socket"
    device: str = "cuda:0"

    process_window: float = 240.0
    process_hop: float = 20.0
    raw_buffer_seconds: float = 260.0
    stft_win_seconds: float = 20.0
    stft_hop_seconds: float = 1.0
    doa_delay: float = 5.0
    doa_win_len: float = 10.0
    doa_mode: str = "center"

    frequency_resolution: int = 20
    f_lower_bound: float = 0.0
    f_higher_bound: float = 400.0
    denoise_thresh: float = 0.1
    line_channel: int = 0
    spec_freq_div: float = 20.0
    add_f_lower_bound: bool = False

    track_match_max_dt: float = 240.0
    track_match_max_df: float = 1.0
    analysis_window: float = 1200.0
    debug_dir: str = None

    @classmethod
    def default(cls, project_root=None, device="cuda:0"):
        root = os.path.abspath(project_root or os.getcwd())
        model_path = os.path.join(
            root,
            "work_dir",
            "yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05",
            "yolo_deepDenoiser_20_240.pth",
        )
        return cls(project_root=root, model_path=model_path, device=device)

    def resolved_model_path(self):
        if os.path.isabs(self.model_path):
            return self.model_path
        return os.path.join(self.project_root, self.model_path)

    def apply_mapping(self, data: Mapping[str, Any]) -> None:
        """Apply model/algorithm settings from a service config mapping."""
        if not data:
            return

        known_fields = {f.name for f in fields(self)}
        model_data = data.get("model") or {}
        algorithm_data = data.get("algorithm") or {}
        flat_data = {
            key: value
            for key, value in data.items()
            if key in known_fields
        }

        if not isinstance(model_data, Mapping):
            raise ValueError("config.model must be an object")
        if not isinstance(algorithm_data, Mapping):
            raise ValueError("config.algorithm must be an object")

        for key, value in flat_data.items():
            self._set_config_value(key, value)

        model_aliases = {
            "path": "model_path",
            "model_path": "model_path",
            "processor_module": "processor_module",
            "project_root": "project_root",
            "device": "device",
        }
        for key, value in model_data.items():
            target = model_aliases.get(key)
            if target is None:
                raise ValueError("unsupported model config key: %s" % key)
            self._set_config_value(target, value)

        for key, value in algorithm_data.items():
            if key not in known_fields:
                raise ValueError("unsupported algorithm config key: %s" % key)
            self._set_config_value(key, value)

    def _set_config_value(self, key: str, value: Any) -> None:
        current = getattr(self, key)
        setattr(self, key, _coerce_like(value, current))

    def validate(self):
        if not os.path.isdir(self.project_root):
            raise FileNotFoundError("project_root does not exist: %s" % self.project_root)
        model_path = self.resolved_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError("model_path does not exist: %s" % model_path)
        if self.process_hop <= 0:
            raise ValueError("process_hop must be positive")
        if self.stft_hop_seconds <= 0:
            raise ValueError("stft_hop_seconds must be positive")
        if self.spec_freq_div <= 0:
            raise ValueError("spec_freq_div must be positive")
