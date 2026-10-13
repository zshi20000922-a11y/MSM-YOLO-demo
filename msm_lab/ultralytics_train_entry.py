"""Subprocess entry point for native Ultralytics detection training."""

from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> int:
    raw = os.environ.get("MSM_ULTRALYTICS_TRAIN_REQUEST", "")
    if not raw:
        print("MSM_ULTRALYTICS_TRAIN_REQUEST is missing", file=sys.stderr)
        return 2
    try:
        request = json.loads(raw)
        from ultralytics import YOLO

        options = dict(request.get("options") or {})
        name = str(options.pop("name", "train"))
        if request.get("resume"):
            model = YOLO(request["model"])
            model.train(resume=True)
        else:
            options.pop("resume", None)
            options["data"] = request["dataset"]
            options["project"] = request["project"]
            options["name"] = name
            YOLO(request["model"]).train(**options)
        return 0
    except KeyboardInterrupt:
        print("Training interrupted by user; Ultralytics received SIGINT.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
