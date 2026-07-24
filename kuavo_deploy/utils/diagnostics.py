import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import tarfile
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_SECRET_PATHS = (
    Path("/opt/kuavo_secrets/kuavo_diag.env"),
    Path("/root/kuavo_data_challenge/secrets/kuavo_diag.env"),
    Path("secrets/kuavo_diag.env"),
)


def _truthy(value: Optional[str]) -> bool:
    token = str(value or "").strip().lower().split(maxsplit=1)[0] if str(value or "").strip() else ""
    return token in {"1", "true", "yes", "on"}


def _redact_env_value(key: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if any(secret in key.upper() for secret in ("SECRET", "TOKEN", "KEY", "URL")):
        return "<set>"
    return str(value)


def _load_env_file(path: Path) -> Dict[str, str]:
    values = {}
    try:
        exists = path.exists()
    except OSError:
        return values
    if not exists:
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _jsonable(value: Any) -> Any:
    np = None
    try:
        import numpy as _np

        np = _np
    except Exception:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bgr_uint8(value: Any):
    import numpy as np

    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    if not isinstance(value, np.ndarray):
        return None
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {1, 3, 4}:
        value = value.transpose(1, 2, 0)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3:
        return None
    if value.dtype != np.uint8:
        value = value.astype(np.float32, copy=False)
        if value.size and float(np.nanmax(value)) <= 1.5:
            value = value * 255.0
        value = np.nan_to_num(value, nan=0.0, posinf=255.0, neginf=0.0)
        value = np.clip(value, 0, 255).astype(np.uint8)
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=2)
    if value.shape[-1] < 3:
        return None
    return np.ascontiguousarray(value[..., :3][..., ::-1])


def _select_observation_image_keys(
    observation: Dict[str, Any],
    requested_csv: str = "",
    default_head_only: bool = False,
    max_keys: Optional[int] = None,
) -> List[str]:
    image_keys = [str(key) for key in observation if "images" in str(key)]
    requested = [value.strip() for value in str(requested_csv or "").split(",") if value.strip()]
    if requested:
        selected = []
        for key in image_keys:
            if "all" in requested or any(key == value or key.endswith("." + value) for value in requested):
                selected.append(key)
    elif default_head_only:
        selected = []
        for key in image_keys:
            if key.endswith(".head_cam_h"):
                selected = [key]
                break
        if not selected:
            selected = image_keys[:1]
    else:
        selected = list(image_keys)
    if max_keys is not None and max_keys > 0:
        selected = selected[:max_keys]
    return selected


class NullEpisodeRecorder:
    def log_action(self, *args, **kwargs):
        return None

    def log_state(self, *args, **kwargs):
        return None

    def log_timing(self, *args, **kwargs):
        return None

    def finalize(self, *args, **kwargs):
        return None


class AsyncObservationVideoRecorder:
    """Best-effort observation video writer that never blocks the control loop."""

    _STOP = object()

    def __init__(
        self,
        manager: "DiagnosticsManager",
        output_directory: Path,
        episode: int,
        observation: Dict[str, Any],
        fps: Any,
    ):
        self.manager = manager
        self.output_directory = Path(output_directory)
        self.episode = episode
        self.fps = self._safe_fps(fps)
        self.keys = self._select_keys(observation)
        self.paths = {
            key: self.output_directory / f"rollout_{episode}_{self._short_key(key)}.mp4"
            for key in self.keys
        }
        self._queue = queue.Queue(maxsize=2)
        self._closed = False
        self._thread = None
        if self.keys:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def submit(self, observation: Any) -> None:
        if self._closed or not self.keys or not isinstance(observation, dict):
            return
        try:
            frames = {key: observation[key] for key in self.keys if key in observation}
            if frames:
                self._queue.put_nowait(frames)
        except queue.Full:
            return
        except Exception:
            self.manager.logger.debug("failed to enqueue diagnostics video frame", exc_info=True)

    def close(self, timeout_sec: float = 10.0) -> List[Path]:
        if self._closed:
            return self._completed_paths()
        self._closed = True
        if self._thread is None:
            return []
        try:
            while True:
                try:
                    self._queue.put_nowait(self._STOP)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
            self._thread.join(timeout=max(0.0, timeout_sec))
        except Exception:
            self.manager.logger.debug("failed to close diagnostics video recorder", exc_info=True)
        return self._completed_paths()

    def _run(self) -> None:
        writers = {}
        try:
            import cv2

            self.output_directory.mkdir(parents=True, exist_ok=True)
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                for key, value in item.items():
                    try:
                        frame = self._to_bgr_uint8(value)
                        if frame is None:
                            continue
                        height, width = frame.shape[:2]
                        writer = writers.get(key)
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(self.paths[key]),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                self.fps,
                                (width, height),
                            )
                            if not writer.isOpened():
                                writer.release()
                                continue
                            writers[key] = writer
                        writer.write(frame)
                    except Exception:
                        self.manager.logger.debug("failed to write diagnostics video frame", exc_info=True)
        except Exception:
            self.manager.logger.debug("diagnostics video worker failed", exc_info=True)
        finally:
            for writer in writers.values():
                try:
                    writer.release()
                except Exception:
                    pass

    def _select_keys(self, observation: Dict[str, Any]) -> List[str]:
        return _select_observation_image_keys(
            observation,
            requested_csv=self.manager.env.get("KUAVO_DIAG_VIDEO_KEYS", ""),
            default_head_only=True,
            max_keys=None,
        )

    def _to_bgr_uint8(self, value: Any):
        return _to_bgr_uint8(value)

    def _completed_paths(self) -> List[Path]:
        return sorted(path for path in self.paths.values() if path.exists() and path.stat().st_size > 0)

    def _safe_fps(self, fps: Any) -> float:
        try:
            return max(1.0, float(fps))
        except (TypeError, ValueError):
            return 10.0

    def _short_key(self, key: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key.rsplit(".", 1)[-1])


class AsyncObservationSnapshotUploader:
    """Best-effort periodic JPEG uploader that keeps inference non-blocking."""

    _STOP = object()

    def __init__(
        self,
        manager: "DiagnosticsManager",
        episode: int,
        observation: Dict[str, Any],
        recorder: Optional["EpisodeDiagnosticsRecorder"] = None,
        interval_sec: Any = None,
        max_cameras: Any = None,
    ):
        self.manager = manager
        self.episode = episode
        self.recorder = recorder
        self.interval_sec = max(
            0.5,
            _safe_float(
                interval_sec
                if interval_sec is not None
                else manager.env.get("KUAVO_DIAG_SNAPSHOT_INTERVAL_SEC", "5"),
                5.0,
            ),
        )
        self.max_cameras = max(
            1,
            _safe_int(
                max_cameras
                if max_cameras is not None
                else manager.env.get("KUAVO_DIAG_SNAPSHOT_MAX_CAMERAS", "4"),
                4,
            ),
        )
        self.keys = _select_observation_image_keys(
            observation,
            requested_csv=manager.env.get("KUAVO_DIAG_SNAPSHOT_KEYS", "all"),
            default_head_only=False,
            max_keys=self.max_cameras,
        )
        self.manager.logger.debug(
            "diagnostics snapshot uploader configured: episode=%s interval_sec=%.3f "
            "max_cameras=%s selected_keys=%s observation_keys=%s",
            self.episode,
            self.interval_sec,
            self.max_cameras,
            self.keys,
            list(observation.keys()) if isinstance(observation, dict) else None,
        )
        self._queue = queue.Queue(maxsize=2)
        self._closed = False
        self._last_submit_time = 0.0
        self._thread = None
        if self.keys:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self.manager.logger.debug("diagnostics snapshot uploader disabled for episode=%s: no image keys", episode)

    def submit(self, observation: Any, step: Optional[int] = None, force: bool = False) -> None:
        if self._closed or not self.keys or not isinstance(observation, dict):
            return
        now = time.time()
        if not force and now - self._last_submit_time < self.interval_sec:
            return
        self._last_submit_time = now
        try:
            frames = {key: observation[key] for key in self.keys if key in observation}
            if frames:
                self._queue.put_nowait({"time": now, "step": step, "frames": frames})
                self.manager.logger.debug(
                    "enqueued diagnostics snapshot: episode=%s step=%s keys=%s force=%s",
                    self.episode,
                    step,
                    list(frames.keys()),
                    force,
                )
            else:
                self.manager.logger.debug(
                    "skipped diagnostics snapshot enqueue: episode=%s step=%s selected_keys=%s observation_keys=%s",
                    self.episode,
                    step,
                    self.keys,
                    list(observation.keys()) if isinstance(observation, dict) else None,
                )
        except queue.Full:
            self.manager.logger.debug("dropped diagnostics snapshot: queue full episode=%s step=%s", self.episode, step)
            return
        except Exception:
            self.manager.logger.debug("failed to enqueue diagnostics snapshot", exc_info=True)

    def close(self, timeout_sec: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return
        try:
            while True:
                try:
                    self._queue.put_nowait(self._STOP)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
            self._thread.join(timeout=max(0.0, timeout_sec))
        except Exception:
            self.manager.logger.debug("failed to close diagnostics snapshot uploader", exc_info=True)

    def _run(self) -> None:
        try:
            import cv2

            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                try:
                    uploaded = self._upload_snapshot_item(cv2, item)
                    if uploaded:
                        self._upload_live_diagnostics(item)
                except Exception:
                    self.manager.logger.debug("diagnostics snapshot upload failed", exc_info=True)
        except Exception:
            self.manager.logger.debug("diagnostics snapshot worker failed", exc_info=True)

    def _upload_snapshot_item(self, cv2: Any, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        bucket = self.manager._make_bucket()
        uploaded = []
        wall_time = _safe_float(item.get("time"), time.time())
        step = item.get("step")
        stamp = datetime.datetime.utcfromtimestamp(wall_time).strftime("%Y%m%dT%H%M%S.%fZ")
        for key, value in item.get("frames", {}).items():
            try:
                frame = _to_bgr_uint8(value)
                if frame is None:
                    continue
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue
                object_key = self.manager._object_key(
                    "/".join(
                        [
                            "runs",
                            self.manager.run_id,
                            f"episode_{self.episode:03d}",
                            "snapshots",
                            f"step_{int(step) if step is not None else -1:06d}_{stamp}_{self._short_key(key)}.jpg",
                        ]
                    )
                )
                self.manager.logger.debug(
                    "uploading diagnostics snapshot camera: episode=%s step=%s camera=%s object_key=%s",
                    self.episode,
                    step,
                    key,
                    object_key,
                )
                result = bucket.put_object(object_key, encoded.tobytes())
                uploaded.append({"camera": key, "object_key": object_key, "status": result.status})
            except Exception:
                self.manager.logger.debug("failed to upload diagnostics snapshot camera", exc_info=True)
        if uploaded:
            self.manager.logger.debug(
                "uploaded diagnostics snapshot: episode=%s step=%s uploaded=%s",
                self.episode,
                step,
                uploaded,
            )
            self.manager._send_serverchan_async(
                "Kuavo snapshot uploaded",
                self.manager._format_details(
                    {
                        "run_id": self.manager.run_id,
                        "episode": self.episode,
                        "step": step,
                        "cameras": [item["camera"] for item in uploaded],
                        "object_keys": [item["object_key"] for item in uploaded],
                    }
                ),
            )
        else:
            self.manager.logger.debug(
                "diagnostics snapshot produced no uploads: episode=%s step=%s frame_keys=%s",
                self.episode,
                step,
                list(item.get("frames", {}).keys()),
            )
        return uploaded

    def _upload_live_diagnostics(self, item: Dict[str, Any]) -> None:
        if self.recorder is None:
            return
        step = item.get("step")
        wall_time = _safe_float(item.get("time"), time.time())
        stamp = datetime.datetime.utcfromtimestamp(wall_time).strftime("%Y%m%dT%H%M%S.%fZ")
        try:
            bucket = self.manager._make_bucket()
            uploaded = []
            for name, path in self.recorder.live_paths().items():
                try:
                    path = Path(path)
                    if not path.exists() or path.stat().st_size <= 0:
                        continue
                    object_key = self.manager._object_key(
                        "/".join(
                            [
                                "runs",
                                self.manager.run_id,
                                f"episode_{self.episode:03d}",
                                "live",
                                f"step_{int(step) if step is not None else -1:06d}_{stamp}_{name}",
                            ]
                        )
                    )
                    self.manager.logger.debug(
                        "uploading live diagnostics file: episode=%s step=%s name=%s path=%s object_key=%s",
                        self.episode,
                        step,
                        name,
                        path,
                        object_key,
                    )
                    result = bucket.put_object_from_file(object_key, str(path))
                    uploaded.append({"name": name, "object_key": object_key, "status": result.status})
                except Exception:
                    self.manager.logger.debug("failed to upload live diagnostics file: %s", name, exc_info=True)
            if uploaded:
                self.manager.logger.debug(
                    "uploaded live diagnostics files: episode=%s step=%s uploaded=%s",
                    self.episode,
                    step,
                    uploaded,
                )
            else:
                self.manager.logger.debug(
                    "live diagnostics produced no uploads: episode=%s step=%s",
                    self.episode,
                    step,
                )
        except Exception:
            self.manager.logger.debug("failed to upload live diagnostics", exc_info=True)

    def _short_key(self, key: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key.rsplit(".", 1)[-1])


class EpisodeDiagnosticsRecorder:
    def __init__(self, manager: "DiagnosticsManager", episode: int, config: Any):
        self.manager = manager
        self.episode = episode
        self.config = config
        self.episode_dir = manager.diagnostics_dir / f"episode_{episode:03d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path = self.episode_dir / "actions.jsonl"
        self.states_path = self.episode_dir / "states.jsonl"
        self.timings_path = self.episode_dir / "timings.jsonl"
        self.manifest_path = self.episode_dir / "manifest.json"
        self._lock = threading.Lock()

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        try:
            row.setdefault("episode", self.episode)
            row.setdefault("wall_time", time.time())
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        except Exception:
            self.manager.logger.debug("failed to write diagnostics jsonl", exc_info=True)

    def log_action(self, step: int, action: Any, infer_time_sec: Optional[float] = None) -> None:
        self._append_jsonl(
            self.actions_path,
            {
                "step": step,
                "action": action,
                "infer_time_sec": infer_time_sec,
            },
        )

    def log_state(self, step: int, state: Any, reward: Any = None) -> None:
        self._append_jsonl(
            self.states_path,
            {
                "step": step,
                "observation_state": state,
                "reward": reward,
            },
        )

    def log_timing(self, step: int, **timings: Any) -> None:
        row = {"step": step}
        row.update(timings)
        self._append_jsonl(self.timings_path, row)

    def live_paths(self) -> Dict[str, Path]:
        return {
            "actions.jsonl": self.actions_path,
            "states.jsonl": self.states_path,
            "timings.jsonl": self.timings_path,
        }

    def finalize(
        self,
        video_paths: Iterable[Path],
        success: bool,
        steps: int,
        fps: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        try:
            video_paths = [Path(p) for p in video_paths if p]
            manifest = self._build_manifest(video_paths, success, steps, fps, extra or {})
            self.manifest_path.write_text(
                json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            bundle_path = self.manager.bundles_dir / f"episode_{self.episode:03d}_{self.manager.run_id}.tar.gz"
            with tarfile.open(str(bundle_path), "w:gz") as tar:
                for path in (self.manifest_path, self.actions_path, self.states_path, self.timings_path):
                    if path.exists():
                        tar.add(str(path), arcname=f"episode_{self.episode:03d}/{path.name}")
                for video_path in video_paths:
                    if video_path.exists():
                        tar.add(
                            str(video_path),
                            arcname=f"episode_{self.episode:03d}/videos/{video_path.name}",
                        )
            return bundle_path
        except Exception:
            self.manager.logger.debug("failed to finalize diagnostics bundle", exc_info=True)
            return None

    def _build_manifest(
        self,
        video_paths: List[Path],
        success: bool,
        steps: int,
        fps: Any,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg_env = getattr(self.config, "env", None)
        cfg_inf = getattr(self.config, "inference", None)
        manifest = {
            "run_id": self.manager.run_id,
            "episode": self.episode,
            "generated_at": datetime.datetime.now().isoformat(),
            "success": bool(success),
            "steps": int(steps),
            "fps": _jsonable(fps),
            "videos": [str(p) for p in video_paths],
            "run_meta": self.manager.run_meta,
            "env": {
                "env_name": getattr(cfg_env, "env_name", None),
                "which_arm": getattr(cfg_env, "which_arm", None),
                "eef_type": getattr(cfg_env, "eef_type", None),
                "ros_rate": getattr(cfg_env, "ros_rate", None),
                "image_size": getattr(cfg_env, "image_size", None),
                "frame_alignment": getattr(cfg_env, "frame_alignment", None),
                "ratio": getattr(cfg_env, "ratio", None),
                "obs_keys": list(getattr(cfg_env, "obs_key_map", {}).keys()),
            },
            "inference": {
                "policy_type": getattr(cfg_inf, "policy_type", None),
                "task": getattr(cfg_inf, "task", None),
                "method": getattr(cfg_inf, "method", None),
                "timestamp": getattr(cfg_inf, "timestamp", None),
                "epoch": getattr(cfg_inf, "epoch", None),
                "max_episode_steps": getattr(cfg_inf, "max_episode_steps", None),
            },
        }
        manifest.update(extra)
        return manifest


class DiagnosticsManager:
    def __init__(self, output_directory: Path, run_meta: Dict[str, Any], env: Dict[str, str]):
        self.output_directory = Path(output_directory)
        self.run_meta = dict(run_meta)
        self.env = env
        self.enabled = _truthy(env.get("KUAVO_DIAG_ENABLED"))
        self.smoke_test_enabled = _truthy(env.get("KUAVO_DIAG_SMOKE_TEST", "1"))
        self.snapshot_enabled = _truthy(env.get("KUAVO_DIAG_SNAPSHOT_ENABLED", "1"))
        self.serverchan_url = env.get("SERVERCHAN_SEND_URL", "").strip()
        self.oss_endpoint = env.get("OSS_ENDPOINT", "").strip()
        self.oss_bucket = env.get("OSS_BUCKET", "").strip()
        self.oss_prefix = env.get("OSS_PREFIX", "kuavo-diagnostics").strip().strip("/")
        self.access_key_id = env.get("OSS_ACCESS_KEY_ID", "").strip()
        self.access_key_secret = env.get("OSS_ACCESS_KEY_SECRET", "").strip()
        self.serverchan_connect_timeout = max(
            0.2,
            _safe_float(env.get("KUAVO_DIAG_SERVERCHAN_CONNECT_TIMEOUT_SEC", "1.0"), 1.0),
        )
        self.serverchan_read_timeout = max(
            0.2,
            _safe_float(env.get("KUAVO_DIAG_SERVERCHAN_READ_TIMEOUT_SEC", "2.0"), 2.0),
        )
        self.run_id = self._make_run_id(run_meta)
        self.diagnostics_dir = self.output_directory / "diagnostics"
        self.bundles_dir = self.diagnostics_dir / "bundles"
        self.logger = self._setup_file_logger(self.diagnostics_dir / "diagnostics_upload.log")
        try:
            self.bundles_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.enabled = False
            self.logger.debug("failed to create diagnostics directory; diagnostics disabled", exc_info=True)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if self.enabled else None
        self._futures = []
        self._notified_started = False
        self.logger.debug(
            "diagnostics initialized: enabled=%s smoke_test_enabled=%s snapshot_enabled=%s "
            "run_id=%s output_directory=%s oss_bucket=%s oss_prefix=%s serverchan_configured=%s "
            "env_source=%s raw_diag_enabled=%s raw_snapshot_enabled=%s raw_snapshot_interval=%s raw_snapshot_keys=%s",
            self.enabled,
            self.smoke_test_enabled,
            self.snapshot_enabled,
            self.run_id,
            self.output_directory,
            self.oss_bucket,
            self.oss_prefix,
            bool(self.serverchan_url),
            self.env.get("KUAVO_DIAG_ENV_SOURCE"),
            _redact_env_value("KUAVO_DIAG_ENABLED", self.env.get("KUAVO_DIAG_ENABLED")),
            _redact_env_value("KUAVO_DIAG_SNAPSHOT_ENABLED", self.env.get("KUAVO_DIAG_SNAPSHOT_ENABLED")),
            _redact_env_value("KUAVO_DIAG_SNAPSHOT_INTERVAL_SEC", self.env.get("KUAVO_DIAG_SNAPSHOT_INTERVAL_SEC")),
            _redact_env_value("KUAVO_DIAG_SNAPSHOT_KEYS", self.env.get("KUAVO_DIAG_SNAPSHOT_KEYS")),
        )

    @classmethod
    def from_env(
        cls,
        output_directory: Path,
        run_meta: Dict[str, Any],
        secret_paths: Iterable[Path] = DEFAULT_SECRET_PATHS,
    ) -> "DiagnosticsManager":
        merged = dict(os.environ)
        loaded_path = None
        for path in secret_paths:
            values = _load_env_file(Path(path))
            if values:
                merged.update(values)
                loaded_path = str(path)
                break
        merged["KUAVO_DIAG_ENV_SOURCE"] = loaded_path or "process-environment"
        return cls(output_directory=output_directory, run_meta=run_meta, env=merged)

    def notify_eval_started(self) -> None:
        if not self.enabled or self._notified_started:
            self.logger.debug(
                "skipped eval started notification: enabled=%s already_notified=%s",
                self.enabled,
                self._notified_started,
            )
            return
        self._notified_started = True
        details = {
            "run_id": self.run_id,
            "output_directory": str(self.output_directory),
            "oss_bucket": self.oss_bucket,
            "oss_prefix": self.oss_prefix,
        }
        details.update(self.run_meta)
        self.logger.debug("sending eval started notification: %s", details)
        self._send_serverchan_async("Kuavo eval started", self._format_details(details))

    def smoke_test_async(self) -> None:
        if not self.enabled or not self.smoke_test_enabled:
            return
        self._submit(self._smoke_test)

    def start_episode(self, episode: int, config: Any) -> Any:
        if not self.enabled:
            return NullEpisodeRecorder()
        try:
            return EpisodeDiagnosticsRecorder(self, episode, config)
        except Exception:
            self.logger.debug("failed to start episode diagnostics recorder", exc_info=True)
            return NullEpisodeRecorder()

    def start_video_recorder(
        self,
        output_directory: Path,
        episode: int,
        observation: Dict[str, Any],
        fps: Any,
    ) -> Optional[AsyncObservationVideoRecorder]:
        if not self.enabled:
            return None
        try:
            return AsyncObservationVideoRecorder(self, output_directory, episode, observation, fps)
        except Exception:
            self.logger.debug("failed to start diagnostics video recorder", exc_info=True)
            return None

    def start_snapshot_uploader(
        self,
        episode: int,
        observation: Dict[str, Any],
        recorder: Optional[EpisodeDiagnosticsRecorder] = None,
        interval_sec: Any = None,
        max_cameras: Any = None,
    ) -> Optional[AsyncObservationSnapshotUploader]:
        if not self.enabled or not self.snapshot_enabled:
            self.logger.debug(
                "skipped diagnostics snapshot uploader: enabled=%s snapshot_enabled=%s",
                self.enabled,
                self.snapshot_enabled,
            )
            return None
        try:
            return AsyncObservationSnapshotUploader(
                self,
                episode=episode,
                observation=observation,
                recorder=recorder,
                interval_sec=interval_sec,
                max_cameras=max_cameras,
            )
        except Exception:
            self.logger.debug("failed to start diagnostics snapshot uploader", exc_info=True)
            return None

    def upload_async(self, bundle_path: Optional[Path], episode: int, success: bool) -> None:
        if not self.enabled or bundle_path is None:
            return
        self._submit(self._upload_bundle, Path(bundle_path), episode, success)

    def close(self, timeout_sec: float = 60.0) -> None:
        if self._executor is None:
            return
        deadline = time.time() + timeout_sec
        for future in list(self._futures):
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0:
                break
            try:
                future.result(timeout=remaining)
            except Exception:
                self.logger.debug("diagnostics task failed during close", exc_info=True)
        self._executor.shutdown(wait=False)

    def _submit(self, fn, *args):
        if self._executor is None:
            return None
        try:
            future = self._executor.submit(fn, *args)
            self._futures.append(future)
            return future
        except Exception:
            self.logger.debug("failed to submit diagnostics task", exc_info=True)
            return None

    def _smoke_test(self) -> None:
        try:
            bucket = self._make_bucket()
            key = self._object_key(f"_smoke_test/{self.run_id}_{int(time.time())}.txt")
            result = bucket.put_object(key, b"kuavo diagnostics oss smoke test\n")
            self._send_serverchan_async(
                "Kuavo OSS smoke test ok",
                self._format_details({"run_id": self.run_id, "status": result.status, "object_key": key}),
            )
        except Exception as exc:
            self.logger.debug("OSS smoke test failed: %s", exc, exc_info=True)
            self._send_serverchan_async(
                "Kuavo OSS smoke test failed",
                self._format_details({"run_id": self.run_id, "error": self._brief_error()}),
            )

    def _upload_bundle(self, bundle_path: Path, episode: int, success: bool) -> None:
        try:
            bucket = self._make_bucket()
            bundle_size = bundle_path.stat().st_size
            bundle_sha256 = _sha256_file(bundle_path)
            key = self._object_key(f"runs/{self.run_id}/episode_{episode:03d}/{bundle_path.name}")
            result = bucket.put_object_from_file(key, str(bundle_path))
            self._send_serverchan_async(
                "Kuavo episode uploaded",
                self._format_details(
                    {
                        "run_id": self.run_id,
                        "episode": episode,
                        "success": bool(success),
                        "status": result.status,
                        "object_key": key,
                        "size_bytes": bundle_size,
                        "sha256": bundle_sha256,
                    }
                ),
            )
        except Exception:
            self.logger.debug("failed to upload diagnostics bundle", exc_info=True)
            self._send_serverchan_async(
                "Kuavo episode upload failed",
                self._format_details(
                    {
                        "run_id": self.run_id,
                        "episode": episode,
                        "success": bool(success),
                        "bundle_path": str(bundle_path),
                        "error": self._brief_error(),
                    }
                ),
            )

    def _make_bucket(self):
        import oss2

        missing = [
            name
            for name, value in (
                ("OSS_ACCESS_KEY_ID", self.access_key_id),
                ("OSS_ACCESS_KEY_SECRET", self.access_key_secret),
                ("OSS_ENDPOINT", self.oss_endpoint),
                ("OSS_BUCKET", self.oss_bucket),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("missing diagnostics OSS config: " + ", ".join(missing))
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        return oss2.Bucket(auth, self.oss_endpoint, self.oss_bucket)

    def _send_serverchan_async(self, title: str, desp: str) -> None:
        if not self.serverchan_url:
            self.logger.debug("serverchan url missing; skipped notification: %s", title)
            return
        try:
            thread = threading.Thread(target=self._send_serverchan, args=(title, desp), daemon=True)
            thread.start()
        except Exception:
            self.logger.debug("failed to start serverchan notification thread: %s", title, exc_info=True)

    def _send_serverchan(self, title: str, desp: str) -> None:
        if not self.serverchan_url:
            self.logger.debug("serverchan url missing; skipped notification: %s", title)
            return
        try:
            import requests

            response = requests.post(
                self.serverchan_url,
                data={"title": title, "desp": desp},
                timeout=(self.serverchan_connect_timeout, self.serverchan_read_timeout),
            )
            body = (response.text or "")[:500]
            if response.status_code >= 400:
                self.logger.debug(
                    "serverchan notification returned error: title=%s status=%s body=%s",
                    title,
                    response.status_code,
                    body,
                )
            else:
                self.logger.debug(
                    "serverchan notification sent: title=%s status=%s body=%s",
                    title,
                    response.status_code,
                    body,
                )
        except Exception:
            self.logger.debug("failed to send serverchan notification: %s", title, exc_info=True)

    def _object_key(self, suffix: str) -> str:
        prefix = self.oss_prefix.strip("/")
        suffix = suffix.strip("/")
        return f"{prefix}/{suffix}" if prefix else suffix

    def _format_details(self, details: Dict[str, Any]) -> str:
        lines = []
        for key, value in details.items():
            lines.append(f"- {key}: `{_jsonable(value)}`")
        return "\n".join(lines)

    def _brief_error(self) -> str:
        text = traceback.format_exc(limit=3).strip()
        return text[-1800:] if len(text) > 1800 else text

    def _make_run_id(self, run_meta: Dict[str, Any]) -> str:
        parts = [
            str(run_meta.get("task") or "task"),
            str(run_meta.get("method") or "method"),
            str(run_meta.get("timestamp") or "timestamp"),
            "epoch" + str(run_meta.get("epoch") or "unknown"),
            _utc_stamp(),
        ]
        safe = []
        for part in parts:
            safe.append("".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in part))
        return "_".join(safe)

    def _setup_file_logger(self, path: Path) -> logging.Logger:
        logger = logging.getLogger(f"kuavo_diagnostics.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if not logger.handlers:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(str(path), encoding="utf-8")
            except Exception:
                handler = logging.NullHandler()
            handler.setLevel(logging.DEBUG)
            if hasattr(handler, "setFormatter"):
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger
