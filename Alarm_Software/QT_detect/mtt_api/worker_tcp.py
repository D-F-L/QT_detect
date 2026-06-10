import argparse
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .models import AudioChunk, ProcessingResult
from .pipeline import MarineAlgorithmPipeline
from .config import ProcessorConfig, load_config_file


_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=7, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


class ProtocolError(Exception):
    pass


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> Tuple[dict, bytes]:
    prefix = _read_exact(sock, 8)
    header_len, payload_len = struct.unpack(">II", prefix)
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError("invalid header length: %d" % header_len)
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError("payload too large: %d" % payload_len)

    header_raw = _read_exact(sock, header_len)
    payload = _read_exact(sock, payload_len) if payload_len else b""
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError("invalid json header: %s" % exc) from exc
    if not isinstance(header, dict):
        raise ProtocolError("frame header must be a json object")
    return header, payload


def write_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header.setdefault("protocol", PROTOCOL_VERSION)
    header_raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header_raw) > MAX_HEADER_BYTES:
        raise ProtocolError("response header too large: %d" % len(header_raw))
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("response payload too large: %d" % len(payload))
    sock.sendall(struct.pack(">II", len(header_raw), len(payload)) + header_raw + payload)


def _audio_dtype(header: dict) -> np.dtype:
    dtype = str(header.get("dtype", "float32")).lower()
    if dtype not in ("float32", "f4"):
        raise ProtocolError("unsupported audio dtype: %s" % dtype)

    byte_order = str(header.get("byte_order", "little")).lower()
    if byte_order in ("little", "le", "<"):
        return np.dtype("<f4")
    if byte_order in ("big", "be", ">"):
        return np.dtype(">f4")
    raise ProtocolError("unsupported byte_order: %s" % byte_order)


def decode_audio_chunk(header: dict, payload: bytes) -> AudioChunk:
    buoy_id = str(header.get("buoy_id") or "")
    if not buoy_id:
        raise ProtocolError("audio_chunk missing buoy_id")

    fs = int(header.get("fs", 0))
    if fs <= 0:
        raise ProtocolError("audio_chunk invalid fs: %s" % header.get("fs"))

    samples = int(header.get("samples", 0))
    if samples <= 0:
        samples = len(payload) // (3 * 4)
    expected_bytes = samples * 3 * 4
    if len(payload) != expected_bytes:
        raise ProtocolError("audio payload size mismatch: got %d expected %d" % (len(payload), expected_bytes))

    channels = header.get("channels") or ["pt", "vx", "vy"]
    if [str(c).lower() for c in channels] != ["pt", "vx", "vy"]:
        raise ProtocolError("unsupported channel order: %s" % channels)

    arr = np.frombuffer(payload, dtype=_audio_dtype(header))
    if arr.size != samples * 3:
        raise ProtocolError("audio sample count mismatch")
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().newbyteorder()
    arr = np.asarray(arr, dtype=np.float32)

    start_time = header.get("start_time")
    if start_time is None:
        start_time_ms = header.get("start_time_ms")
        start_time = float(start_time_ms) / 1000.0 if start_time_ms is not None else time.time()

    return AudioChunk(
        buoy_id=buoy_id,
        fs=fs,
        start_time=float(start_time),
        pt=arr[0:samples].copy(),
        vx=arr[samples:samples * 2].copy(),
        vy=arr[samples * 2:samples * 3].copy(),
    )


def result_has_payload(result: ProcessingResult) -> bool:
    idae = result.idae
    recognition = result.recognition
    return (
        idae.rows > 0
        and idae.cols > 0
        and (idae.power_db is not None or idae.noisy_spec is not None or idae.denoise_spec is not None)
    ) or bool(idae.line_points or idae.trajectories or recognition.clusters or recognition.cluster_analysis)


@dataclass
class PipelineEntry:
    fs: int
    process: mp.Process
    input_queue: mp.Queue
    output_queue: mp.Queue
    lock: threading.Lock


def _worker_loop(
    config: ProcessorConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    include_array_data: bool,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
):
    _setup_logging(log_file, log_level)
    _wlog = logging.getLogger(__name__ + ".worker")

    pipeline: Optional[MarineAlgorithmPipeline] = None
    while True:
        item = input_queue.get()
        if item is None:
            break

        seq = item.get("seq")
        try:
            chunk = decode_audio_chunk(item["header"], item["payload"])
            if pipeline is None:
                pipeline = MarineAlgorithmPipeline(config)
                _wlog.info("pipeline initialized  buoy=%s", chunk.buoy_id)

            result = pipeline.push_audio(chunk, include_array_data=include_array_data)
            if not result_has_payload(result):
                output_queue.put({
                    "type": "no_result",
                    "seq": seq,
                    "buoy_id": chunk.buoy_id,
                    "status": "ok",
                    "relative_end_time": result.relative_end_time,
                })
                continue

            body = json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            _wlog.info("processing_result  seq=%s  buoy=%s", seq, chunk.buoy_id)
            output_queue.put({
                "type": "processing_result",
                "seq": seq,
                "buoy_id": chunk.buoy_id,
                "status": "ok",
                "payload_encoding": "json",
                "content_type": "application/json",
                "payload": body,
            })
        except Exception as exc:
            _wlog.error("worker error  seq=%s  buoy=%s: %s", seq,
                        item.get("header", {}).get("buoy_id"), exc, exc_info=True)
            output_queue.put({
                "type": "error",
                "seq": seq,
                "buoy_id": item.get("header", {}).get("buoy_id"),
                "status": "error",
                "message": str(exc),
            })


class PipelineRegistry:
    def __init__(
        self,
        base_config: ProcessorConfig,
        include_array_data: bool,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        self._base_config = base_config
        self._include_array_data = include_array_data
        self._log_file = log_file
        self._log_level = log_level
        self._entries: Dict[str, PipelineEntry] = {}
        self._lock = threading.Lock()
        self._mp_context = mp.get_context("spawn")

    def get(self, buoy_id: str, fs: int) -> PipelineEntry:
        with self._lock:
            entry = self._entries.get(buoy_id)
            if entry is not None and entry.fs == fs and entry.process.is_alive():
                return entry
            if entry is not None:
                self._stop_entry(entry)

            input_queue = self._mp_context.Queue(maxsize=4)
            output_queue = self._mp_context.Queue(maxsize=4)
            process = self._mp_context.Process(
                target=_worker_loop,
                args=(self._base_config, input_queue, output_queue,
                      self._include_array_data, self._log_file, self._log_level),
                daemon=True,
                name="mtt-worker-%s" % buoy_id,
            )
            process.start()
            entry = PipelineEntry(
                fs=fs,
                process=process,
                input_queue=input_queue,
                output_queue=output_queue,
                lock=threading.Lock(),
            )
            self._entries[buoy_id] = entry
            return entry

    def shutdown(self):
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._stop_entry(entry)

    @staticmethod
    def _stop_entry(entry: PipelineEntry):
        try:
            entry.input_queue.put_nowait(None)
        except Exception:
            pass
        entry.process.join(timeout=3)
        if entry.process.is_alive():
            entry.process.terminate()
            entry.process.join(timeout=2)


class MarineWorkerTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, registry: PipelineRegistry, include_array_data: bool):
        super().__init__(server_address, handler_cls)
        self.registry = registry
        self.include_array_data = include_array_data


_hlog = logging.getLogger(__name__ + ".handler")


class MarineWorkerHandler(socketserver.BaseRequestHandler):
    request: socket.socket
    server: MarineWorkerTcpServer

    def handle(self):
        peer = "%s:%s" % self.client_address
        _hlog.info("client connected: %s", peer)
        while True:
            try:
                header, payload = read_frame(self.request)
            except EOFError:
                break
            except Exception as exc:
                self._send_error(None, None, str(exc))
                break

            msg_type = header.get("type")
            seq = header.get("seq")
            buoy_id = header.get("buoy_id")
            try:
                if msg_type == "hello":
                    write_frame(self.request, {
                        "type": "hello_ack",
                        "seq": seq,
                        "status": "ok",
                        "server": "mtt_api.worker_tcp",
                    })
                elif msg_type == "audio_chunk":
                    self._handle_audio(seq, header, payload)
                elif msg_type == "stop":
                    write_frame(self.request, {"type": "stopped", "seq": seq, "status": "ok"})
                    break
                else:
                    raise ProtocolError("unsupported message type: %s" % msg_type)
            except Exception as exc:
                _hlog.warning("protocol error  seq=%s  buoy=%s: %s", seq, buoy_id, exc)
                self._send_error(seq, buoy_id, str(exc))

        _hlog.info("client disconnected: %s", peer)

    def _handle_audio(self, seq, header: dict, payload: bytes):
        buoy_id = str(header.get("buoy_id") or "")
        fs = int(header.get("fs", 0))
        if not buoy_id:
            raise ProtocolError("audio_chunk missing buoy_id")
        if fs <= 0:
            raise ProtocolError("audio_chunk invalid fs: %s" % header.get("fs"))
        entry = self.server.registry.get(buoy_id, fs)

        write_frame(self.request, {
            "type": "ack",
            "seq": seq,
            "buoy_id": buoy_id,
            "status": "received",
        })

        with entry.lock:
            entry.input_queue.put({
                "seq": seq,
                "header": header,
                "payload": payload,
            })
            response = entry.output_queue.get()

        response_type = response.pop("type", "error")
        response_payload = response.pop("payload", b"")
        write_frame(self.request, {"type": response_type, **response}, response_payload)

    def _send_error(self, seq: Optional[int], buoy_id: Optional[str], message: str):
        try:
            write_frame(self.request, {
                "type": "error",
                "seq": seq,
                "buoy_id": buoy_id,
                "status": "error",
                "message": message,
            })
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarineTargetTracker Python algorithm TCP server")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="JSON/YAML config file for server, runtime, model, and algorithm settings")
    parser.add_argument("--host", default=None, help="TCP listen host")
    parser.add_argument("--port", type=int, default=None, help="TCP listen port")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-module", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-array-data", action="store_true", default=None,
                        help="Return matrix shapes only, without base64 data")
    parser.add_argument("--log-file", default=None, metavar="PATH",
                        help="Write logs to this file (rotated daily, kept 7 days). Default: stdout only.")
    parser.add_argument("--log-level", default=None,
                        choices=["debug", "info", "warning", "error"],
                        help="Log level")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    config_doc = load_config_file(args.config) if args.config else {}
    server_doc = config_doc.get("server") or {}
    runtime_doc = config_doc.get("runtime") or {}
    logging_doc = config_doc.get("logging") or {}
    model_doc = config_doc.get("model") or {}

    if not isinstance(server_doc, dict):
        raise ValueError("config.server must be an object")
    if not isinstance(runtime_doc, dict):
        raise ValueError("config.runtime must be an object")
    if not isinstance(logging_doc, dict):
        raise ValueError("config.logging must be an object")
    if not isinstance(model_doc, dict):
        raise ValueError("config.model must be an object")

    host = args.host or server_doc.get("host") or "127.0.0.1"
    port = args.port if args.port is not None else int(server_doc.get("port", 18888))
    project_root = (
        args.project_root
        or model_doc.get("project_root")
        or config_doc.get("project_root")
        or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    device = args.device or runtime_doc.get("device") or model_doc.get("device") or "cpu"
    log_file = args.log_file or logging_doc.get("file")
    log_level_name = args.log_level or logging_doc.get("level") or "info"
    log_level = _LOG_LEVELS.get(str(log_level_name).lower(), logging.INFO)
    _setup_logging(log_file, log_level)

    config = ProcessorConfig.default(project_root, device=device)
    config.apply_mapping(config_doc)
    if args.model_path:
        config.model_path = args.model_path
    if args.processor_module:
        config.processor_module = args.processor_module
    if args.device:
        config.device = args.device

    include_array_data = _as_bool(runtime_doc.get("include_array_data", True))
    if args.no_array_data is True:
        include_array_data = False
    registry = PipelineRegistry(config, include_array_data, log_file=log_file, log_level=log_level)

    with MarineWorkerTcpServer((host, port), MarineWorkerHandler, registry, include_array_data) as server:
        logger.info("mtt_api.worker_tcp listening on %s:%d", host, port)
        logger.info("project_root=%s  processor=%s  device=%s",
                    config.project_root, config.processor_module, config.device)
        logger.info("algorithm: process_window=%s process_hop=%s freq=[%s,%s] denoise_thresh=%s",
                    config.process_window, config.process_hop,
                    config.f_lower_bound, config.f_higher_bound,
                    config.denoise_thresh)
        if args.config:
            logger.info("config file: %s", os.path.abspath(args.config))
        if log_file:
            logger.info("log file: %s", os.path.abspath(log_file))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("interrupted, shutting down")
        finally:
            registry.shutdown()


if __name__ == "__main__":
    main()
