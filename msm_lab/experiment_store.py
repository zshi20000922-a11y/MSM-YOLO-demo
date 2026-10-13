from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    packages = ("numpy", "opencv-python", "torch", "ultralytics", "lap", "PySide6")
    result = {"python": sys.version.split()[0], "opencv": cv2.__version__}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ExperimentStore:
    """Versioned local SQLite index; large media/model files stay on disk."""

    SCHEMA_VERSION = 3

    def __init__(self, database: str | Path):
        self.database = Path(database).expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL REFERENCES projects(id),
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    task TEXT NOT NULL DEFAULT 'detect',
                    framework_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, sha256, backend)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id INTEGER NOT NULL REFERENCES projects(id),
                    model_id INTEGER REFERENCES models(id),
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    video_path TEXT NOT NULL,
                    video_sha256 TEXT,
                    model_path TEXT,
                    model_sha256 TEXT,
                    detector_backend TEXT,
                    tracker_backend TEXT,
                    tracker_version TEXT,
                    motion_algorithms_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    device TEXT,
                    summary_json TEXT,
                    output_dir TEXT NOT NULL,
                    output_video TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_project_started
                    ON experiments(project_id, started_at DESC);
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ended_at TEXT,
                    backend TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    dataset_yaml TEXT NOT NULL,
                    dataset_sha256 TEXT,
                    model_source TEXT NOT NULL,
                    model_source_sha256 TEXT,
                    repo_root TEXT NOT NULL,
                    python_executable TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    epoch INTEGER,
                    epochs INTEGER,
                    progress REAL NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    log_tail TEXT NOT NULL DEFAULT '',
                    best_checkpoint TEXT,
                    last_checkpoint TEXT,
                    results_csv TEXT,
                    artifacts_json TEXT NOT NULL DEFAULT '[]',
                    return_code INTEGER,
                    error TEXT,
                    model_id INTEGER REFERENCES models(id)
                );
                CREATE INDEX IF NOT EXISTS idx_training_runs_started
                    ON training_runs(started_at DESC);
                CREATE TABLE IF NOT EXISTS deployment_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    onnx_path TEXT,
                    rknn_path TEXT,
                    error TEXT,
                    log_tail TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_deployment_runs_started
                    ON deployment_runs(started_at DESC);
                PRAGMA user_version = 3;
            """)

    def ensure_project(self, name: str) -> int:
        clean_name = name.strip() or "默认项目"
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO projects(name, created_at) VALUES (?, ?)",
                (clean_name, datetime.now(timezone.utc).isoformat()),
            )
            return int(db.execute(
                "SELECT id FROM projects WHERE name = ?", (clean_name,)
            ).fetchone()[0])

    def register_model(self, project_id: int, path: str | Path,
                       sha256: str, backend: str,
                       framework_version: str, task: str = "detect") -> int:
        model_path = str(Path(path).expanduser().resolve())
        with self._connect() as db:
            db.execute("""
                INSERT OR IGNORE INTO models(
                    project_id, path, sha256, backend, task,
                    framework_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (project_id, model_path, sha256, backend, task,
                  framework_version, datetime.now(timezone.utc).isoformat()))
            row = db.execute("""
                SELECT id FROM models
                WHERE project_id=? AND sha256=? AND backend=?
            """, (project_id, sha256, backend)).fetchone()
            return int(row[0])

    def start_experiment(
        self, *, experiment_id: str, project_id: int, mode: str,
        video_path: str | Path, video_sha256: str | None,
        model_id: int | None, model_path: str | None, model_sha256: str | None,
        detector_backend: str | None, tracker_backend: str | None,
        tracker_version: str | None, motion_algorithms: dict,
        parameters: dict, device: str | None, output_dir: str | Path,
    ) -> None:
        with self._connect() as db:
            db.execute("""
                INSERT INTO experiments(
                    id, project_id, model_id, mode, status, started_at,
                    video_path, video_sha256, model_path, model_sha256,
                    detector_backend, tracker_backend, tracker_version,
                    motion_algorithms_json, parameters_json, runtime_json,
                    device, output_dir
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, project_id, model_id, mode,
                datetime.now(timezone.utc).isoformat(),
                str(Path(video_path).expanduser().resolve()), video_sha256,
                model_path, model_sha256, detector_backend, tracker_backend,
                tracker_version, _json(motion_algorithms), _json(parameters),
                _json(runtime_versions()), device,
                str(Path(output_dir).expanduser().resolve()),
            ))

    def finish_experiment(self, experiment_id: str, *, status: str,
                          summary: dict | None = None,
                          output_video: str | Path | None = None,
                          error: str | None = None) -> None:
        allowed = {"completed", "stopped", "failed"}
        if status not in allowed:
            raise ValueError(f"实验最终状态无效：{status}")
        with self._connect() as db:
            db.execute("""
                UPDATE experiments
                SET status=?, ended_at=?, summary_json=?, output_video=?, error=?
                WHERE id=?
            """, (
                status, datetime.now(timezone.utc).isoformat(),
                _json(summary) if summary is not None else None,
                str(Path(output_video).resolve()) if output_video else None,
                error, experiment_id,
            ))

    def update_experiment_backends(self, experiment_id: str, *,
                                   detector_backend: str | None = None,
                                   tracker_version: str | None = None) -> None:
        with self._connect() as db:
            db.execute("""
                UPDATE experiments
                SET detector_backend=COALESCE(?, detector_backend),
                    tracker_version=COALESCE(?, tracker_version)
                WHERE id=?
            """, (detector_backend, tracker_version, experiment_id))

    def bind_experiment_model(self, experiment_id: str, *, model_id: int,
                              model_path: str | Path, model_sha256: str,
                              detector_backend: str) -> None:
        with self._connect() as db:
            db.execute("""
                UPDATE experiments
                SET model_id=?, model_path=?, model_sha256=?, detector_backend=?
                WHERE id=?
            """, (model_id, str(Path(model_path).expanduser().resolve()),
                  model_sha256, detector_backend, experiment_id))

    def recent_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT e.*, p.name AS project_name
                FROM experiments e JOIN projects p ON p.id=e.project_id
                ORDER BY e.started_at DESC LIMIT ?
            """, (max(1, int(limit)),)).fetchall()
        return [dict(row) for row in rows]

    def create_training_run(self, *, run_id: str, request: dict[str, Any],
                            dataset_sha256: str | None,
                            model_source_sha256: str | None,
                            output_dir: str | Path,
                            environment: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("""
                INSERT INTO training_runs(
                    id, status, started_at, updated_at, backend, request_json,
                    environment_json, dataset_yaml, dataset_sha256,
                    model_source, model_source_sha256, repo_root,
                    python_executable, output_dir, epochs
                ) VALUES (?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status='running', updated_at=excluded.updated_at,
                    ended_at=NULL, request_json=excluded.request_json,
                    environment_json=excluded.environment_json,
                    output_dir=excluded.output_dir, error=NULL,
                    return_code=NULL
            """, (
                run_id, now, now, request.get("backend_id", "spinosaurus_yolov5"),
                _json(request), _json(environment or {}),
                str(Path(request.get("dataset", "")).expanduser().resolve()),
                dataset_sha256,
                str(Path(request.get("model", "")).expanduser().resolve()),
                model_source_sha256,
                str(Path(request.get("repo_root", "")).expanduser().resolve()),
                str(Path(request.get("python_executable", "")).expanduser().resolve()),
                str(Path(output_dir).expanduser().resolve()),
                int(request.get("options", {}).get("epochs", 0)) or None,
            ))

    def update_training_run(self, run_id: str, *, status: str,
                            epoch: int | None = None, epochs: int | None = None,
                            progress: float = 0.0,
                            metrics: dict[str, Any] | None = None,
                            log_tail: str | None = None,
                            artifacts: list[str] | tuple[str, ...] | None = None,
                            return_code: int | None = None,
                            error: str | None = None,
                            model_id: int | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        terminal = status in {"completed", "stopped", "failed"}
        with self._connect() as db:
            db.execute("""
                UPDATE training_runs SET
                    status=?, updated_at=?, ended_at=CASE WHEN ? THEN ? ELSE ended_at END,
                    epoch=COALESCE(?, epoch), epochs=COALESCE(?, epochs),
                    progress=?, metrics_json=COALESCE(?, metrics_json),
                    log_tail=COALESCE(?, log_tail),
                    artifacts_json=COALESCE(?, artifacts_json),
                    best_checkpoint=COALESCE(?, best_checkpoint),
                    last_checkpoint=COALESCE(?, last_checkpoint),
                    results_csv=COALESCE(?, results_csv),
                    return_code=COALESCE(?, return_code), error=COALESCE(?, error),
                    model_id=COALESCE(?, model_id)
                WHERE id=?
            """, (
                status, now, int(terminal), now,
                epoch, epochs, max(0.0, min(1.0, float(progress))),
                _json(metrics) if metrics is not None else None,
                log_tail,
                _json(list(artifacts)) if artifacts is not None else None,
                next((item for item in (artifacts or ()) if item.endswith("/best.pt")), None),
                next((item for item in (artifacts or ()) if item.endswith("/last.pt")), None),
                next((item for item in (artifacts or ()) if item.endswith("/results.csv")), None),
                return_code, error, model_id, run_id,
            ))

    def training_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def recent_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_deployment_run(self, *, run_id: str, request: dict[str, Any],
                              output_dir: str | Path,
                              onnx_path: str | Path,
                              rknn_path: str | Path) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("""
                INSERT INTO deployment_runs(
                    id, status, started_at, updated_at, request_json,
                    output_dir, onnx_path, rknn_path
                ) VALUES (?, 'running', ?, ?, ?, ?, ?, ?)
            """, (
                run_id, now, now, _json(request),
                str(Path(output_dir).expanduser().resolve()),
                str(Path(onnx_path).expanduser().resolve()),
                str(Path(rknn_path).expanduser().resolve()),
            ))

    def update_deployment_run(self, run_id: str, *, status: str,
                              log_tail: str | None = None,
                              error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("""
                UPDATE deployment_runs
                SET status=?, updated_at=?, log_tail=COALESCE(?, log_tail),
                    error=COALESCE(?, error)
                WHERE id=?
            """, (status, now, log_tail, error, run_id))

    def recent_deployment_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM deployment_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]
