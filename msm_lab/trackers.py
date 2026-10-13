from __future__ import annotations

from collections.abc import Callable


def _ultralytics_bytetrack(confidence: float, frame_rate: float,
                           track_buffer: int):
    from msm_yolo_demo import ByteTrackAdapter

    return ByteTrackAdapter(confidence, frame_rate, track_buffer)


TRACKER_BACKENDS: dict[str, tuple[str, Callable[..., object]]] = {
    "bytetrack_ultralytics": (
        "ByteTrack · Ultralytics 8.4.117", _ultralytics_bytetrack,
    ),
}


def tracker_backend_choices() -> list[tuple[str, str]]:
    return [(backend_id, value[0])
            for backend_id, value in TRACKER_BACKENDS.items()]


def create_tracker(backend_id: str, confidence: float,
                   frame_rate: float, track_buffer: int):
    try:
        _, factory = TRACKER_BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(f"未知追踪后端：{backend_id}") from exc
    return factory(confidence, frame_rate, track_buffer)
